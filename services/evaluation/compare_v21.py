#!/usr/bin/env python3
"""AURORA v2.1 — three-way model comparison (Base vs adapter2 vs adapter-v2.1).

Runs the *same* evaluation rows and refusal probes through **two** gateway
instances and merges them into a single three-way report:

* ``--gateway-adapter2``   — gateway running with ``artifacts/finetune/v2/adapter``
                             (exposes ``peft`` = adapter2, ``peft-base`` = base)
* ``--gateway-v21``        — gateway running with ``artifacts/finetune/v2.1/adapter``
                             (exposes ``peft`` = adapter-v2.1, ``peft-base`` = base)

The base anchor (``peft-base``) is the same Base model in both gateways, so the
three provider slots compared are: base, adapter2, adapter-v2.1.

Two report *tracks* are produced and kept clearly separated:

1. ``report-<mode>-model.json`` — ``--evaluate-model`` **ON**: the deterministic
   policy/missing-data application safety gate is bypassed, so this measures the
   **raw** behaviour each model actually learned (how the refusals/grounding
   improved at model level).
2. ``report-<mode>-prod.json`` — ``--evaluate-model`` **OFF**: the application
   safety gate is fully enabled for every request, i.e. the real production
   AURORA behaviour (refusal enforced deterministically at the app layer).

Usage (two terminals):
    # terminal 1 — adapter2
    AURORA_GATEWAY_PEFT_ADAPTER_DIR=../../artifacts/finetune/v2/adapter \\
         ../../.venv/bin/python -m uvicorn app.main:app --port 8102
    # terminal 2 — adapter-v2.1 (after pushing the Colab-trained adapter)
    AURORA_GATEWAY_PEFT_ADAPTER_DIR=../../artifacts/finetune/v2.1/adapter \\
         ../../.venv/bin/python -m uvicorn app.main:app --port 8103

    python services/evaluation/compare_v21.py \\
        --gateway-adapter2 http://localhost:8102 \\
        --gateway-v21      http://localhost:8103 \\
        --rows artifacts/dataset/v2/test.jsonl \\
        --out artifacts/eval/v2-threeway \\
        --mode smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/evaluation"))
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

import model_eval  # noqa: E402

import compare_v2 as cv2  # noqa: E402

LABELS = ["base", "adapter2", "adapter-v2.1"]

# (provider slot) -> (gateway url key, gateway provider name to call)
# base comes from the adapter2 gateway's peft-base; both gateways share the same
# Base weights, so the anchor is identical either way.
SLOT_MAP = {
    "base": ("gw2", "peft-base"),
    "adapter2": ("gw2", "peft"),
    "adapter-v2.1": ("gw21", "peft"),
}


def probe_to_request_for_probe(probe: dict, evaluate_model: bool = False):
    """Build the canonical request a probe *should* be served as (empty telemetry)."""
    return cv2.probe_to_request(probe, "peft", cache_key="none", evaluate_model=evaluate_model)


async def _slot_request(slot: str, req_builder, data, *, run_id: str, evaluate_model: bool):
    """Build the per-slot GatewayRequest for one row/probe."""
    gw_key, provider = SLOT_MAP[slot]
    return req_builder(
        data, provider,
        cache_key=f"v21:{run_id}:{slot}:{data.get('id') or data.get('probe_id', data.get('input', ''))[:24]}",
        evaluate_model=evaluate_model,
    )


async def run_threeway(
    gateways: dict[str, str],
    rows: list[dict],
    *,
    run_id: str,
    include_probes: bool,
    evaluate_model: bool,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """Return (report, hallucinations, refusals, prompt_equivalence, verify_errors).

    Phase 1 — **verify prompt byte-equivalence up front, before any inference**:
    renders the expected canonical prompt (normalizer + context_prompt renderer)
    and diffs it against each gateway's actual /v1/debug/prompt output for every
    slot and row. Any missing/failed or byte-mismatched render aborts the run
    before a single completion is scored.

    Phase 2 — score completions only after every prompt check passed.
    ``report[slot]`` mirrors compare_v2's shape: {"rows": [...], "probe_outcomes": [...]}.
    """
    report: dict = {p: {"rows": [], "probe_outcomes": []} for p in LABELS}
    hallucinations: list[dict] = []
    refusals: list[dict] = []
    prompt_equivalence: list[dict] = []
    verify_errors: list[dict] = []

    async def prompt_of(client, req) -> str | None:
        return (await cv2.fetch_prompt(client, req) or {}).get("prompt")

    async with httpx.AsyncClient(base_url=gateways["gw2"], timeout=90.0) as c2, \
               httpx.AsyncClient(base_url=gateways["gw21"], timeout=90.0) as c21:
        # -------- Phase 1: verify prompt byte-equivalence for every row+slot --------
        for row in rows:
            expected = cv2.expected_prompt(
                cv2.row_to_request(row, "peft", cache_key="none", evaluate_model=evaluate_model)
            )
            row_eq: dict = {}
            for slot in LABELS:
                gw_key, gw_provider = SLOT_MAP[slot]
                client = c2 if gw_key == "gw2" else c21
                req = await _slot_request(
                    slot, cv2.row_to_request, row,
                    run_id=run_id, evaluate_model=evaluate_model,
                )
                rendered = await prompt_of(client, req)
                ok = rendered is not None and rendered == expected
                row_eq[slot] = {"rendered": rendered, "matches_expected": ok}
                prompt_equivalence.append({
                    "row_id": row.get("id"),
                    "provider": slot,
                    "matches_expected": ok,
                    "sha": cv2.sha(rendered or ""),
                })
                if not ok:
                    verify_errors.append({
                        "phase": "row_prompt",
                        "row_id": row.get("id"),
                        "slot": slot,
                        "problem": "rendered is None" if rendered is None else "byte-mismatch vs canonical expected",
                        "expected_sha": cv2.sha(expected),
                        "rendered_sha": cv2.sha(rendered or ""),
                    })
            report.setdefault("prompt_equivalence_summary", []).append({
                "row_id": row.get("id"),
                "expected_sha": cv2.sha(expected),
                "base_sha": cv2.sha(row_eq["base"]["rendered"] or ""),
                "adapter2_sha": cv2.sha(row_eq["adapter2"]["rendered"] or ""),
                "adapter_v21_sha": cv2.sha(row_eq["adapter-v2.1"]["rendered"] or ""),
                "byte_identical_all": (
                    row_eq["base"]["matches_expected"]
                    and row_eq["adapter2"]["matches_expected"]
                    and row_eq["adapter-v2.1"]["matches_expected"]
                ),
                "matches_expected": all(s["matches_expected"] for s in row_eq.values()),
            })

        if include_probes:
            for probe in cv2.REFUSAL_PROBES:
                expected = cv2.expected_prompt(probe_to_request_for_probe(probe, evaluate_model))
                for slot in LABELS:
                    gw_key, _ = SLOT_MAP[slot]
                    client = c2 if gw_key == "gw2" else c21
                    req = await _slot_request(
                        slot, cv2.probe_to_request, probe,
                        run_id=run_id, evaluate_model=evaluate_model,
                    )
                    rendered = await prompt_of(client, req)
                    ok = rendered is not None and rendered == expected
                    prompt_equivalence.append({
                        "row_id": probe.get("id"),
                        "provider": slot,
                        "probe": True,
                        "matches_expected": ok,
                        "sha": cv2.sha(rendered or ""),
                    })
                    if not ok:
                        verify_errors.append({
                            "phase": "probe_prompt",
                            "row_id": probe.get("id"),
                            "slot": slot,
                            "problem": "rendered is None" if rendered is None else "byte-mismatch vs canonical expected",
                            "expected_sha": cv2.sha(expected),
                            "rendered_sha": cv2.sha(rendered or ""),
                        })

        if verify_errors:
            # Do NOT score anything: prompts from the two gateways are not
            # byte-identical to the canonical render, so any model comparison
            # would measure prompt skew, not model quality.
            return report, hallucinations, refusals, prompt_equivalence, verify_errors

        # -------- Phase 2: score completions (each slot seen once per row) --------
        for row in rows:
            for slot in LABELS:
                gw_key, gw_provider = SLOT_MAP[slot]
                client = c2 if gw_key == "gw2" else c21
                req = await _slot_request(
                    slot, cv2.row_to_request, row,
                    run_id=run_id, evaluate_model=evaluate_model,
                )
                comp = await cv2.fetch_completion(client, req)
                content = (comp or {}).get("content", "") or ""
                metrics = model_eval.score_row(
                    content,
                    row.get("context", {}),
                    row.get("derived", {}),
                    row.get("output", ""),
                )
                metrics.pop("answer_quality", None)
                metrics.pop("judge_grounding", None)
                record = cv2.hallucination_case(row, slot, content, row.get("output", ""))
                if record:
                    hallucinations.append(record)
                report[slot]["rows"].append({
                    "row_id": row.get("id"),
                    "scenario": row.get("scenario"),
                    "intent": row.get("intent"),
                    "question": row.get("input"),
                    "expected": row.get("output", ""),
                    "response": content,
                    "refused": bool((comp or {}).get("refused")),
                    "metrics": metrics,
                })

        if include_probes:
            for probe in cv2.REFUSAL_PROBES:
                for slot in LABELS:
                    gw_key, _ = SLOT_MAP[slot]
                    client = c2 if gw_key == "gw2" else c21
                    req = await _slot_request(
                        slot, cv2.probe_to_request, probe,
                        run_id=run_id, evaluate_model=evaluate_model,
                    )
                    comp = await cv2.fetch_completion(client, req)
                    content = (comp or {}).get("content", "") or ""
                    out = cv2.refusal_outcome(probe, slot, content, True)
                    report[slot]["probe_outcomes"].append(out)
                    refusals.append(out)

    return report, hallucinations, refusals, prompt_equivalence, verify_errors


def aggregate_report(report: dict) -> dict:
    agg = cv2.aggregate_report(report, LABELS)
    return {"base": agg["base"], "adapter2": agg["adapter2"], "adapter-v2.1": agg["adapter-v2.1"]}


def hallucination_stats(report: dict, hallucinations: list[dict], n_rows: int) -> dict:
    """Per-slot hallucination counts + rates, and the individually logged cases."""
    stats: dict = {}
    counts = {p: 0 for p in LABELS}
    for h in hallucinations:
        counts[h["provider"]] = counts.get(h["provider"], 0) + 1
    for slot in LABELS:
        rate = report[slot].get("rows", []) and round(
            counts[slot] / n_rows, 4
        ) if n_rows else 0.0
        # Rate over rows that actually have ground truth (defensible denominator)
        stats[slot] = {
            "hallucination_cases": counts[slot],
            "hallucination_rate_of_rows": rate,
            "mean_hallucination_rate": agg_val(report, slot, "hallucination_rate"),
        }
    return stats


def agg_val(report: dict, slot: str, metric: str) -> float | None:
    rows = [r["metrics"] for r in report[slot].get("rows", [])]
    if not rows:
        return None
    a = model_eval.aggregate(rows)
    return a.get(f"mean.{metric}")


def scenario_signal_accuracy(report: dict) -> list[dict]:
    """Signal accuracy by scenario, per slot, with the v2.1-vs-adapter2 delta."""
    by_scn: dict[str, dict[str, list]] = {}
    for slot in LABELS:
        for r in report[slot].get("rows", []):
            scn = r.get("scenario") or "unknown"
            by_scn.setdefault(scn, {}).setdefault(slot, []).append(
                r["metrics"].get("signal_accuracy")
                if r["metrics"].get("signal_accuracy") is not None else 0.0
            )
    out = []
    for scn, slots in sorted(by_scn.items()):
        means = {s: (sum(v) / len(v) if v else None) for s, v in slots.items()}
        n = max((len(v) for v in slots.values()), default=0)
        delta = None
        a2, a21 = means.get("adapter2"), means.get("adapter-v2.1")
        if a2 is not None and a21 is not None:
            delta = round(a21 - a2, 3)
        out.append({
            "scenario": scn,
            "n": n,
            "base": round(means.get("base"), 3) if means.get("base") is not None else None,
            "adapter2": round(a2, 3) if a2 is not None else None,
            "adapter-v2.1": round(a21, 3) if a21 is not None else None,
            "delta_v21_vs_adapter2": delta,
            "v21_regresses_vs_adapter2": delta is not None and delta < 0,
        })
    return out


def best_column(agg: dict) -> dict:
    """Which slot wins each metric (direction-aware)."""
    rows = {}
    lower_better = {"hallucination_rate"}
    for slot in agg:
        for key, val in agg[slot].items():
            if key.startswith("mean."):
                metric = key[len("mean."):]
                rows.setdefault(metric, {})
                rows[metric][slot] = val
    out = {}
    for metric, vals in rows.items():
        present = {s: v for s, v in vals.items() if v is not None}
        if not present:
            continue
        if metric in lower_better:
            best = min(present, key=present.get)
        else:
            best = max(present, key=present.get)
        out[metric] = {**vals, "best": best}
    return out


def production_recommendation(agg: dict, scen_rows: list[dict]) -> dict:
    """Final recommendation on which adapter should be the production candidate."""
    sig = {slot: agg[slot].get("mean.signal_accuracy") for slot in LABELS}
    halu = {slot: agg[slot].get("mean.hallucination_rate") for slot in LABELS}
    f1 = {slot: agg[slot].get("mean.lexical_f1") for slot in LABELS}
    regress = [s["scenario"] for s in scen_rows if s.get("v21_regresses_vs_adapter2")]

    reasons = []
    v21_sig = sig["adapter-v2.1"]
    a2_sig = sig["adapter2"]
    if v21_sig is not None and a2_sig is not None:
        if v21_sig > a2_sig:
            reasons.append(f"adapter-v2.1 higher signal_accuracy ({v21_sig} vs {a2_sig})")
        elif v21_sig < a2_sig:
            reasons.append(f"adapter2 higher signal_accuracy ({a2_sig} vs {v21_sig})")
    if halu["adapter-v2.1"] is not None and halu["adapter2"] is not None:
        if halu["adapter-v2.1"] < halu["adapter2"]:
            reasons.append(f"adapter-v2.1 lower hallucination_rate ({halu['adapter-v2.1']} vs {halu['adapter2']})")
        elif halu["adapter-v2.1"] > halu["adapter2"]:
            reasons.append(f"adapter2 lower hallucination_rate ({halu['adapter2']} vs {halu['adapter-v2.1']})")

    # Defensible default: pick by signal_accuracy, tiebreak by hallucination lower-better.
    def score(slot):
        return (sig.get(slot) or 0.0, -(halu.get(slot) or 0.0), f1.get(slot) or 0.0)

    candidate = "adapter2"
    if (sig.get("adapter-v2.1") or 0.0) > (sig.get("adapter2") or 0.0):
        candidate = "adapter-v2.1"
    elif (sig.get("adapter-v2.1") or 0.0) == (sig.get("adapter2") or 0.0):
        if (halu.get("adapter-v2.1") or 0.0) < (halu.get("adapter2") or 0.0):
            candidate = "adapter-v2.1"

    return {
        "candidate": candidate,
        "reasons": reasons,
        "signal_accuracy": sig,
        "hallucination_rate": halu,
        "scenarios_where_v21_regresses_vs_adapter2": regress,
    }


async def main(args: argparse.Namespace) -> int:
    gateways = {"gw2": args.gateway_adapter2, "gw21": args.gateway_v21}
    rows_path = Path(args.rows)
    if not rows_path.exists():
        print(f"✗ rows file not found: {rows_path}")
        return 2

    for name, url in gateways.items():
        if not await cv2._await_ready(url, timeout=args.ready_timeout):
            print(f"✗ gateway not reachable: {name} @ {url}")
            return 2

    all_rows = [json.loads(l) for l in rows_path.open()]
    if args.mode == "smoke":
        rows = cv2.select_smoke(all_rows, per_scenario=args.per_scenario)
    else:
        rows = all_rows
        if args.limit:
            rows = rows[: args.limit]

    tracks = []
    if args.evaluate_model_track:
        tracks.append(("model", True))
    if args.production_track:
        tracks.append(("prod", False))

    for name, evaluate_model in tracks:
        track_label = "evaluate_model" if evaluate_model else "production (gate ON)"
        print(f"\nThree-way comparison [{track_label}]  (mode={args.mode})")
        print(f"  gateways: adapter2={gateways['gw2']}  adapter-v2.1={gateways['gw21']}")
        print(f"  slots: {LABELS}  rows: {len(rows)} + {len(cv2.REFUSAL_PROBES)} probes")

        run_id = uuid.uuid4().hex[:12]
        (report, hallucinations, refusals, prompt_eq, verify_errors) = await run_threeway(
            gateways, rows,
            run_id=run_id,
            include_probes=True,
            evaluate_model=evaluate_model,
        )
        out_dir = Path(args.out) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if verify_errors:
            print(f"  ✗ PROMPT BYTE-EQUIVALENCE FAILED — {len(verify_errors)} error(s); "
                  f"no completions were scored (run {run_id} aborted).")
            (out_dir / f"report-{args.mode}-{name}-verify-errors.json").write_text(
                json.dumps({"verify_errors": verify_errors}, indent=2, default=str)
            )
            continue

        agg = aggregate_report(report)
        eq_ok = all(
            r["byte_identical_all"]
            for r in report.get("prompt_equivalence_summary", [])
        )
        matches_expected = all(r["matches_expected"] for r in prompt_eq)
        scen_rows = scenario_signal_accuracy(report)
        best = best_column(agg)
        halu_stats = hallucination_stats(report, hallucinations, len(rows))
        rec = production_recommendation(agg, scen_rows)

        payload = {
            "run_id": run_id,
            "mode": args.mode,
            "track": name,
            "evaluate_model": evaluate_model,
            "gateways": gateways,
            "rows_file": str(rows_path),
            "rows_evaluated": len(rows),
            "probe_count": len(cv2.REFUSAL_PROBES),
            "providers": LABELS,
            "aggregate": agg,
            "best_by_metric": best,
            "hallucination_stats": halu_stats,
            "hallucination_cases": hallucinations,
            "signal_accuracy_by_scenario": scen_rows,
            "refusal_outcomes": refusals,
            "prompt_byte_equivalence": {
                "checked_rows": len(report.get("prompt_equivalence_summary", [])),
                "identical_between_all": eq_ok,
                "all_match_expected_canonical_render": matches_expected,
            },
            "production_recommendation": rec,
            "timestamp": time.time(),
        }

        (out_dir / f"report-{args.mode}-{name}.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
        # save adapter2 and adapter-v2.1 (and base) results separately
        for slot in LABELS:
            (out_dir / f"report-{args.mode}-{name}-{slot}.json").write_text(
                json.dumps({
                    "run_id": run_id,
                    "mode": args.mode,
                    "track": name,
                    "slot": slot,
                    "rows": report[slot]["rows"],
                    "probe_outcomes": report[slot]["probe_outcomes"],
                }, indent=2, default=str)
            )

        print(f"  track={name}  run={run_id}")
        for slot in LABELS:
            a = agg[slot]
            print(
                f"    {slot:10} signal={a.get('mean.signal_accuracy'):.3f} "
                f"ground={a.get('mean.grounding'):.3f} "
                f"recall={a.get('mean.text_recall'):.3f} "
                f"f1={a.get('mean.lexical_f1'):.3f} "
                f"halu={a.get('mean.hallucination_rate'):.3f} "
                f"refusal={a.get('probe_refusal_rate')}"
            )
        print(f"  hallucinations: {len(hallucinations)}  refusals: {len(refusals)}")
        print(f"  report: {out_dir / f'report-{args.mode}-{name}.json'}")
        print(f"  prompt byte-equivalence identical: {eq_ok}  matches_expected: {matches_expected}")
        print(f"  candidate: {rec['candidate']} ({'; '.join(rec['reasons'])})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-adapter2", default="http://localhost:8102")
    parser.add_argument("--gateway-v21", default="http://localhost:8103")
    parser.add_argument("--rows", default=str(ROOT / "artifacts/dataset/v2/test.jsonl"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--per-scenario", type=int, default=2, help="smoke rows per scenario")
    parser.add_argument("--limit", type=int, default=0, help="full: cap rows (0 = all)")
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument("--out", default=str(ROOT / "artifacts/eval/v2-threeway"))
    parser.add_argument("--production-track", action="store_true",
                        help="also run with the deterministic safety/missing-data gate ON")
    args = parser.parse_args()
    args.evaluate_model_track = True
    sys.exit(asyncio.run(main(args)))