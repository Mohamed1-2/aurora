#!/usr/bin/env python3
"""AURORA evaluation harness CLI.

Runs the golden dataset through the deployed assistant-api with deterministic
replayed telemetry, scores groundedness + provenance + fidelity, gates the
release, and (optionally) registers the accepted skill pack in MLflow.

Usage:
    python run.py                 # default: local assistant on :8100
    python run.py --assistant http://localhost:8100 --json-out report.json
    python run.py --register       # register accepted pack in MLflow
    python run.py --provider azure-openai --model aurora-chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vehicle_ai.version import PROMPT_VERSION, RAG_INDEX_VERSION, __version__  # noqa: E402

from framework.mlflow_client import log_run, package_config  # noqa: E402
from framework.runner import acceptance, run_suite, summarize  # noqa: E402
from vehicle_ai.schemas import EvaluationReport  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ASSISTANT = "http://localhost:8100"


def load_golden() -> dict:
    import yaml

    with (HERE / "datasets/golden.yml").open() as f:
        return yaml.safe_load(f)


def load_snapshot() -> dict:
    with (HERE / "datasets/telemetry_snapshot.json").open() as f:
        return json.load(f)


async def main(args: argparse.Namespace) -> int:
    golden = load_golden()
    snapshot = load_snapshot()

    run_id = uuid.uuid4().hex[:12]
    timeout = time.time() + 40
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < timeout:
            try:
                r = await client.get(f"{args.assistant}/healthz")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                await asyncio.sleep(1.0)
        else:
            print(f"✗ assistant-api not reachable at {args.assistant}")
            return 2

    print(f"\nAURORA evaluation run {run_id}")
    print(f"  intent: ground assistant on DETERMINISTIC replay telemetry")
    print(f"  prompt: {PROMPT_VERSION}  ·  rag index: {RAG_INDEX_VERSION}")
    print(f"  cases : {len(golden['cases'])}\n")

    cases = await run_suite(args.assistant, golden, snapshot, snapshot["vehicle_id"])
    report = EvaluationReport(
        run_id=run_id,
        timestamp=time.time(),
        strategy="golden-set + replay-grounding + groundedness gate",
        prompt_version=PROMPT_VERSION,
        rag_index_version=RAG_INDEX_VERSION,
        provider=args.provider or "auto",
        model=args.model or "auto",
        environment=args.environment,
        aggregate={},
        cases=cases,
    )
    report.aggregate = summarize(report)

    dest = Path(args.json_out if args.json_out else HERE.parent.parent / "artifacts/eval" / run_id)
    dest.mkdir(parents=True, exist_ok=True)
    report_file = dest / "report.json"
    report_file.write_text(report.model_dump_json(indent=2))
    md = _render_markdown(report)
    (dest / "summary.md").write_text(md)

    verdict = acceptance(report, min_pass_rate=args.min_pass_rate)

    # optional MLflow register on acceptance
    mlflow_run = mlflow_model = None
    if args.register:
        pkg = package_config({
            "prompt_version": PROMPT_VERSION,
            "rag_index_version": RAG_INDEX_VERSION,
            "assistant_version": __version__,
            "provider": report.provider,
            "model": report.model,
        })
        pkg_path = dest / "skillpack.json"
        pkg_path.write_bytes(pkg)
        mlflow_run, mlflow_model = log_run(
            tracking_uri=args.mlflow_uri,
            params={"prompt_version": PROMPT_VERSION, "rag_index_version": RAG_INDEX_VERSION, "environment": args.environment},
            metrics=report.aggregate | {"case_score_mean": report.aggregate.get("case_score_mean", 0.0)},
            artifact_paths={"report": str(report_file), "summary": str(dest / "summary.md"), "package": str(pkg_path)},
            register=verdict["passed"],
        )
        report.mlflow_run_id = mlflow_run
        report.mlflow_registered_model = mlflow_model
        report_file.write_text(report.model_dump_json(indent=2))

    print(md)
    status = "✓ ACCEPTED — ship to production" if verdict["passed"] else "✗ REJECTED — improve before rollout"
    print(f"\nGATE: {status}   (pass rate {report.aggregate.get('pass_rate', 0)} ≥ {args.min_pass_rate})")
    print(f"Report: {report_file}")
    if mlflow_run:
        print(f"MLflow run: {mlflow_run}")
        print(f"Registered model: {mlflow_model}")
    print("AI-assisted evaluation: metrics are grounded in vehicle data; a production"
          "LLM-as-judge pass is available when your model provider is configured.\n")
    return 0 if verdict["passed"] else 1


def _render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# AURORA Evaluation · run `{report.run_id}`",
        "",
        f"- strategy: {report.strategy}",
        f"- prompt `{report.prompt_version}` · rag index `{report.rag_index_version}`",
        f"- pass rate: **{report.aggregate.get('pass_rate', 0):.1%}**",
        "",
        "| case | passed | signal_acc | doc_ground | text_recall | grade |",
        "|------|--------|-----------|-----------|-------------|-------|",
    ]
    for c in report.cases:
        m = c.metrics
        lines.append(
            f"| {c.case_id} | {'✓' if c.passed else '✗'} | {m.get('signal_accuracy', '-')} | "
            f"{m.get('doc_grounding', '-')} | {m.get('text_recall', '-')} | {m.get('grade', '-')} |"
        )
    lines.append("")
    lines.append("| aggregate | value |")
    lines.append("|-----------|-------|")
    for k, v in sorted(report.aggregate.items()):
        if k in ("cases_total",):
            continue
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assistant", default=DEFAULT_ASSISTANT)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--environment", default="ci")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))