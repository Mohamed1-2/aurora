"""AURORA Monitoring — drift detection + fleet telemetry health."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from vehicle_ai.logging_conf import configure_logging, get_logger
from vehicle_ai.metrics import MetricsRegistry, metrics_available
from vehicle_ai.version import __version__

from app.config import settings
from app.drift import DriftDetector

configure_logging("monitoring", settings.log_level)
logger = get_logger("aurora.monitoring")
metrics = MetricsRegistry()

_baseline_path = Path(settings.baseline_file)
if not _baseline_path.is_absolute():
    _baseline_path = Path(__file__).resolve().parent.parent / settings.baseline_file

detector = DriftDetector(
    baseline_frames=settings.baseline_frames,
    eval_frames=settings.eval_frames,
    psi_threshold=settings.psi_threshold,
    ks_threshold=settings.ks_threshold,
    baseline_file=str(_baseline_path),
)

async def _ws_listener() -> None:
    import websockets

    while True:
        try:
            async with websockets.connect(settings.telemetry_ws, ping_interval=20, max_size=2**20) as ws:
                async for raw in ws:
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if "signals" in frame:
                        detector.ingest(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("monitor telemetry listener reconnecting", extra={"payload": {"error": str(exc)}})
            await asyncio.sleep(3.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_ws_listener())
    yield
    task.cancel()


app = FastAPI(title="AURORA Monitoring", version=__version__, lifespan=lifespan)


@app.get("/monitor/drift")
def drift():
    report = detector.evaluate()
    for k, v in report.get("per_signal_psi", {}).items():
        metrics.drift_gauge.labels("ORION-X-TEST-01", k).set(v)
    return report


@app.get("/ai-quality")
async def ai_quality():
    """Aggregate AI-quality counters from the fleet's Prometheus endpoints.

    Real deployment: point Prometheus here and read rate()/histogram_quantile().
    This lightweight version scrapes each service and tallies live counters,
    enough to render the platform dashboard without external infra.
    """
    endpoints = _parse_services(settings.services)
    out: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=4.0) as client:
        for name, base in endpoints:
            try:
                text = (await client.get(f"{base}/metrics")).text
                out[name] = _parse_prom(text)
            except httpx.HTTPError:
                out[name] = {"error": "unreachable"}
    totals: dict[str, object] = {}
    for src, vals in out.items():
        for k, v in vals.items():
            totals[k] = totals.get(k, 0) + (v or 0)  # type: ignore[operator]
    return {"per_service": out, "platform": totals}


def _parse_prom(text: str) -> dict[str, float]:
    """Naive text-format parser for the AURORA counters (labels stripped)."""
    import re

    sums: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        m = re.match(r"^(\w+)\{.*\}\s+([0-9eE.+\-]+)", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if name.startswith(("aurora_ai_", "aurora_request_")):
            try:
                sums[name] = sums.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return sums


@app.get("/service-health")
async def service_health():
    out: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, base in (_parse_services(settings.services)):
            try:
                r = await client.get(f"{base}/healthz")
                out[name] = {"up": r.status_code == 200, "status_code": r.status_code, "detail": r.json() if r.status_code == 200 else r.text[:120]}
            except httpx.HTTPError as exc:
                out[name] = {"up": False, "status_code": 0, "detail": str(exc)}
    up = sum(1 for v in out.values() if v["up"])
    return {"total": len(out), "up": up, "degraded": up < len(out), "services": out}


@app.get("/insights")
def insights():
    d = detector.evaluate()
    return {
        "drift": d,
        "observed_signals": len(detector._latest),
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "monitoring", "baseline_source": detector.baseline_source}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "frames": len(detector.frames), "baseline": detector.baseline is not None}


@app.get("/metrics")
def metrics_endpoint():
    if not metrics_available():
        return Response("prometheus disabled", media_type="text/plain")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _parse_services(spec: str) -> list[tuple[str, str]]:
    out = []
    for pair in spec.split(","):
        if ":" not in pair:
            continue
        name, base = pair.split(":", 1)
        out.append((name, base))
    return out


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_config=None)