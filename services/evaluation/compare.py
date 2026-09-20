#!/usr/bin/env python3
"""Head-to-head comparison of a BASELINE vs a FINE-TUNED assistant.

Runs the identical golden set + deterministic replay telemetry against two
assistant endpoints (a baseline and one serving the fine-tuned adapter) and
emits a side-by-side delta table, the pass verdicts, and a markdown report.

This is the *produce-defensible-evidence* step of the fine-tuning loop: CI
blocks on the candidate adapter beating-or-tying the committed baseline on the
same grounded metrics, so we never regress while we improve.

Usage:
    python compare.py \
        --baseline http://localhost:8100 \
        --finetuned http://localhost:8105 \
        [--method lora-8 --register]
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

from framework.runner import acceptance, run_suite, summarize  # noqa: E402
from vehicle_ai.schemas import EvaluationReport  # noqa: E402

HERE = Path(__file__).resolve().parent
DELTA_METRICS = (
    "pass_rate",
    "mean.grade",
    "mean.signal_accuracy",
    "mean.doc_grounding",
    "mean.text_recall",
    "mean.refusal_honesty",
    "mean.policy_compliance",
    "latency.mean_ms",
)

# Metrics where LOWER is better (negative delta = improvement, not a regression).
_LOWER_IS_BETTER = {"latency.mean_ms"}


def _classify_deltas(rows: dict) -> tuple[dict, dict]:
    """Split metric deltas into improvements vs regressions.

    Direction-aware: a negative delta on a lower-is-better metric (latency) is an
    improvement, not a regression. Returns (improved, regressed) dicts.
    """
    improved: dict = {}
    regressed: dict = {}
    for m, d in rows.items():
        if not isinstance(d, (int, float)) or not d:
            continue
        good = d > 0 if m not in _LOWER_IS_BETTER else d < 0
        (improved if good else regressed)[m] = d
    return improved, regressed


def load_golden() -> dict:
    import yaml

    with (HERE / "datasets/golden.yml").open() as f:
        return yaml.safe_load(f)


def load_snapshot() -> dict:
    with (HERE / "datasets/telemetry_snapshot.json").open() as f:
        return json.load(f)


async def _await_ready(assistant: str, timeout: float = 40) -> bool:
    import httpx

    end = time.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < end:
            try:
                if (await client.get(f"{assistant}/healthz")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False


async def _report_for(assistant: str, golden: dict, snapshot: dict, env: str) -> EvaluationReport:
    cases = await run_suite(assistant, golden, snapshot, snapshot["vehicle_id"])
    rep = EvaluationReport(
        run_id=uuid.uuid4().hex[:12],
        timestamp=time.time(),
        strategy="golden-set + replay-grounding (baseline vs finetuned delta)",
        prompt_version="aurora-prompt-v8",
        rag_index_version="rag-index-v4",
        provider="auto",
        model="auto",
        environment=env,
        aggregate={},
        cases=cases,
    )
    rep.aggregate = summarize(rep)
    return rep


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


async def main(args: argparse.Namespace) -> int:
    golden = load_golden()
    snapshot = load_snapshot()

    if not await _await_ready(args.baseline):
        print(f"✗ baseline not reachable at {args.baseline}")
        return 2
    if not await _await_ready(args.finetuned):
        print(f"✗ finetuned not reachable at {args.finetuned}")
        return 2

    print(f"\nAURORA baseline vs fine-tuned comparison ({args.method})")
    print(f"  cases: {len(golden['cases'])}\n")
    baseline = await _report_for(args.baseline, golden, snapshot, "baseline")
    finetuned = await _report_for(args.finetuned, golden, snapshot, "finetuned")

    b, f = baseline.aggregate, finetuned.aggregate
    print(f"{'metric':<24} {'baseline':>10} {'finetuned':>10} {'Δ':>10}")
    print("-" * 56)
    rows = {}
    for m in DELTA_METRICS:
        bv, fv = b.get(m, float("nan")), f.get(m, float("nan"))
        bnum = isinstance(bv, (int, float))
        fnum = isinstance(fv, (int, float))
        delta = fv - bv if bnum and fnum else "n/a"
        rows[m] = delta
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            print(f"{m:<24} {_fmt(bv):>10} {_fmt(fv):>10} {delta:>+10.3f}")
        else:
            print(f"{m:<24} {_fmt(bv):>10} {_fmt(fv):>10} {delta:>10}")

    improved, regressed = _classify_deltas(rows)
    passed = finetuned.aggregate.get("pass_rate", 0) >= baseline.aggregate.get("pass_rate", 0)
    verdict = "IMPLEMENTED" if passed and not regressed else "NEEDS REVIEW"

    dest = HERE.parent.parent / "artifacts/finetune/compare"
    dest.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(baseline, finetuned, rows, verdict, args.method)
    (dest / "baseline-vs-finetuned.md").write_text(md)
    (dest / "metrics.json").write_text(
        json.dumps({
            "method": args.method,
            "baseline": b,
            "finetuned": f,
            "delta": rows,
            "verdict": verdict,
        }, indent=2)
    )

    print(f"\nverdict: {verdict}")
    if improved:
        print(f"  improved: {', '.join(f'{k} {v:+.3f}' for k, v in improved.items())}")
    if regressed:
        print(f"  regressed: {', '.join(f'{k} {v:.3f}' for k, v in regressed.items())}")
    if args.register:
        print("  (MLflow register is handled by run.py --register on the winning artifact.)")
    print(f"  report: {dest / 'baseline-vs-finetuned.md'}")
    return 0 if passed else 1


def _render_markdown(base, finetuned, rows, verdict, method) -> str:
    lines = [
        "# AURORA · Baseline vs Fine-Tuned adapter\n",
        f"- comparison method: `{method}`",
        f"- golden cases: {len(finetuned.cases)} (incl. safety/adversarial)",
        f"- reproduction: `python services/evaluation/compare.py "
        f"--baseline <url-a> --finetuned <url-b>`",
        "",
        "## Delta table",
        "",
        "| metric | baseline | finetuned | Δ |",
        "|--------|----------|-----------|----|",
    ]
    for m in DELTA_METRICS:
        bv = base.aggregate.get(m, "-")
        fv = finetuned.aggregate.get(m, "-")
        d = rows.get(m, "-")
        ds = f"{d:+.3f}" if isinstance(d, (int, float)) and not isinstance(d, bool) else str(d)
        lines.append(f"| {m} | {_fmt(bv)} | {_fmt(fv)} | {ds} |")
    lines += ["", f"**Verdict: {verdict}**", "",
              "> The fine-tuned adapter must beat or tie the committed baseline on the",
              "> same grounded, deterministic golden set before it may ship — evidence, not vibes."]
    return "\n".join(lines)


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline assistant URL")
    parser.add_argument("--finetuned", required=True, help="fine-tuned assistant URL")
    parser.add_argument("--method", default="lora-r8-alpha16")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
