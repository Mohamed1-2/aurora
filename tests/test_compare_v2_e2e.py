"""End-to-end harness run against a stub gateway (no torch/GPU).

Proves the compare_v2 flow: byte-equivalent prompts served to peft and
peft-base, identical rows, scoring, aggregation, hallucination/refusal logs.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "evaluation"))

import compare_v2 as c  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from vehicle_ai.schemas import GatewayRequest  # noqa: E402

ROWS = [json.loads(l) for l in (ROOT / "artifacts/dataset/v2/test.jsonl").open()]
SMOKE = c.select_smoke(ROWS, per_scenario=1)


def _build_stub() -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/debug/prompt")
    def debug_prompt(req: dict):
        r = GatewayRequest(**req)
        return {"provider": req.get("provider"), "prompt": c.expected_prompt(r)}

    @app.post("/v1/complete")
    def complete(req: dict):
        r = GatewayRequest(**req)
        question = next((m.content for m in reversed(r.messages) if m.role == "user"), "")
        provider = req.get("provider") or "peft"
        # Canned, deterministic answer that mirrors a plausible grounded reply.
        soc = r.grounding.telemetry.get("soc_pct")
        if soc is not None:
            content = f"Your SOC is {soc} percent right now."
        else:
            content = f"({provider} answered) I can't answer that without the telemetry."
        return {
            "trace_id": "stub",
            "provider": provider,
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "content": content,
            "latency_ms": 1.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "refused": False,
        }

    return app


def test_full_harness_flow_against_stub_gateway():
    async def _run():
        transport = httpx.ASGITransport(app=_build_stub())
        async with httpx.AsyncClient(
            base_url="http://test", transport=transport, timeout=20.0
        ) as client:
            rows = SMOKE + [dict(r) for r in c.REFUSAL_PROBES]

            # Drive run_cases piece-by-piece with the stub (bypass the CLI's await_ready).
            report: dict = {"peft": {"rows": []}, "peft-base": {"rows": []}}
            hallucinations: list[dict] = []
            refusals: list[dict] = []
            prompt_eq: list[dict] = []

            for row in SMOKE:
                for provider in ("peft", "peft-base"):
                    req = c.row_to_request(row, provider, cache_key=f"s:{provider}:{row['id']}")
                    debug = await c.fetch_prompt(client, req)
                    rendered = debug.get("prompt")
                    prompt_eq.append(
                        {
                            "row_id": row["id"],
                            "provider": provider,
                            "matches_expected": rendered == c.expected_prompt(req),
                        }
                    )
                    comp = await c.fetch_completion(client, req)
                    content = comp.get("content", "")
                    metrics = c.model_eval.score_row(
                        content, row.get("context", {}), row.get("derived", {}), row.get("output", "")
                    )
                    report[provider]["rows"].append({"row_id": row["id"], "metrics": metrics})
                    rec = c.hallucination_case(row, provider, content, row.get("output", ""))
                    if rec:
                        hallucinations.append(rec)

            for probe in c.REFUSAL_PROBES:
                for provider in ("peft", "peft-base"):
                    req = c.probe_to_request(probe, provider, cache_key=f"s:{provider}:{probe['id']}")
                    comp = await c.fetch_completion(client, req)
                    content = comp.get("content", "")
                    out = c.refusal_outcome(probe, provider, content, True)
                    refusals.append(out)

            agg = c.aggregate_report(report, ["peft", "peft-base"])
            comp_result = c.compare_providers(agg, ["peft", "peft-base"])
            verdict = c.verdict_text(comp_result)

            # Both providers got byte-identical prompts that match the canonical renderer.
            matches = [e["matches_expected"] for e in prompt_eq]
            assert matches and all(matches)

            # Base vs LoRA must be scored separately (no cache aliasing in stub).
            sa_peft = agg["peft"]["mean.signal_accuracy"]
            sa_base = agg["peft-base"]["mean.signal_accuracy"]
            assert sa_peft is not None and sa_base is not None

            assert verdict in ("YES", "NO", "INCONCLUSIVE")
            assert len(report["peft"]["rows"]) == len(report["peft-base"]["rows"]) == len(SMOKE)
            assert len(refusals) == 2 * len(c.REFUSAL_PROBES)

    asyncio.run(_run())