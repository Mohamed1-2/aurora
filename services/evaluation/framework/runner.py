"""Evaluation runner: golden set -> assistant -> scores -> report."""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import httpx
import yaml

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.schemas import (
    AssistantResponse,
    CaseResult,
    EvaluationReport,
    PolicyLevel,
)

from .metrics import (
    doc_grounding,
    intent_agreement,
    pxx,
    reference_overlap,
    refusal_check,
    signal_grounding,
)

logger = get_logger("aurora.eval")

HARD_PASS = {"signal_accuracy": 0.5, "doc_grounding": 0.3}

GRADE_WEIGHTS = {"signal_accuracy": 0.45, "doc_grounding": 0.2, "text_recall": 0.25, "intent_match": 0.1}


def _truth(snapshot: dict) -> dict[str, float]:
    return {k: float(v) for k, v in snapshot.get("signals", {}).items() if isinstance(v, (int, float))}


async def run_suite(
    assistant_url: str,
    golden: dict,
    snapshot: dict,
    vehicle_id: str,
) -> list[CaseResult]:
    truth = _truth(snapshot)
    results: list[CaseResult] = []
    async with httpx.AsyncClient(timeout=40.0) as client:
        for case in golden["cases"]:
            try:
                resp = await client.post(
                    f"{assistant_url}/v1/ask",
                    json={"question": case["question"], "vehicle_id": vehicle_id, "ground_telemetry": True, "use_rag": True},
                )
                resp.raise_for_status()
                data: dict = resp.json()
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CaseResult(
                        case_id=case["id"],
                        passed=False,
                        metrics={"gateway": 0.0},
                        notes=[f"assistant call failed: {exc}"],
                    )
                )
                continue

            out = AssistantResponse(**data)
            out_of_scope = bool(case.get("out_of_scope", False))
            expected_policy = case.get("expected_policy")
            expect_refusal = bool(case.get("expect_refusal", False))
            metrics: dict[str, float] = {}

            if out_of_scope or expect_refusal:
                metrics["refusal_honesty"] = refusal_check(out.answer, out.refused)
                policy_ok = True
                if expected_policy is not None:
                    policy_ok = out.policy_level.value == expected_policy
                refusal_policy_ok = True
                if expect_refusal:
                    refusal_policy_ok = out.refused is True
                metrics["policy_compliance"] = 1.0 if (policy_ok and refusal_policy_ok) else 0.0
                passed = metrics["refusal_honesty"] >= 0.5 and metrics["policy_compliance"] >= 1.0
            else:
                policy_ok = True
                if expected_policy is not None:
                    policy_ok = out.policy_level.value == expected_policy
                sig = signal_grounding(
                    out.answer,
                    out.grounding.telemetry_signals,
                    case.get("expected_signals", []),
                    case.get("signal_tolerances", {}),
                    truth,
                )
                doc = doc_grounding(out.citations, case.get("expected_docs", []))
                ov = reference_overlap(out.answer, case.get("reference_answer", ""))
                intent_match = intent_agreement(out.intents, case.get("expected_intents", []))
                metrics["signal_accuracy"] = sig["signal_accuracy"]
                metrics["signal_presence"] = sig["signal_presence"]
                metrics["doc_grounding"] = doc
                metrics["intent_match"] = intent_match
                metrics["policy_compliance"] = 1.0 if policy_ok else 0.0
                metrics.update(ov)
                grade = sum(metrics.get(k, 0.0) * w for k, w in GRADE_WEIGHTS.items())
                hard = all(metrics.get(k, 0.0) >= v for k, v in HARD_PASS.items())
                passed = (grade >= 0.55 and hard and not out.refused) if policy_ok else False
                metrics["grade"] = round(grade, 3)

            # add grade to policy cases too for the aggregate case_score_mean
            if out_of_scope or expect_refusal:
                metrics["grade"] = round(metrics.get("refusal_honesty", 0.0) * (0.5 if not policy_ok else 1.0), 3)

            notes: list[str] = []
            if not passed:
                notes.append("below acceptance gate")
            results.append(
                CaseResult(
                    case_id=case["id"],
                    passed=passed,
                    metrics=metrics,
                    latencies_ms={"assistant_total": out.latency_ms},
                    cost_usd=0.0,
                    citations=out.citations,
                    notes=notes,
                )
            )
    return results


def summarize(report: EvaluationReport) -> dict[str, Any]:
    agg: dict[str, float] = {}
    all_metrics: dict[str, list[float]] = {}
    for c in report.cases:
        for k, v in c.metrics.items():
            all_metrics.setdefault(k, []).append(v)

    for k, vals in all_metrics.items():
        agg[f"mean.{k}"] = round(statistics.fmean(vals), 3)
    lat = [c.latencies_ms.get("assistant_total", 0.0) for c in report.cases if c.latencies_ms]
    if lat:
        agg["latency.p50_ms"] = round(pxx(lat, 0.50))
        agg["latency.p95_ms"] = round(pxx(lat, 0.95))
        agg["latency.mean_ms"] = round(statistics.fmean(lat), 1)
        agg["latency.max_ms"] = round(max(lat), 1)
    passed = sum(1 for c in report.cases if c.passed)
    agg["pass_rate"] = round(passed / len(report.cases), 3) if report.cases else 0.0
    agg["cases_total"] = float(len(report.cases))
    agg["case_score_mean"] = statistics.fmean([c.metrics.get("grade", 0.0) for c in report.cases]) if report.cases else 0.0
    return agg


def acceptance(report: EvaluationReport, min_pass_rate: float = 0.85) -> dict[str, Any]:
    agg = report.aggregate
    ok = agg.get("pass_rate", 0.0) >= min_pass_rate
    return {"passed": ok, "pass_rate": agg.get("pass_rate", 0.0), "min_pass_rate": min_pass_rate}