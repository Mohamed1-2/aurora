"""AURORA Assistant API — the vehicle-facing AI assistant."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from vehicle_ai.logging_conf import configure_logging, get_logger
from vehicle_ai.metrics import MetricsRegistry, metrics_available
from vehicle_ai.version import __version__

from app.api.routes import router
from app.api.operations import router as operations_router, restore_release
from app.config import settings
from app.core.proactive_engine import ProactiveEvaluator
from app.core.service import AssistantPipeline
from app.telemetry.ingest import TelemetryIngest

configure_logging("assistant-api", settings.log_level)
logger = get_logger("aurora.assistant")
metrics = MetricsRegistry()

telemetry = TelemetryIngest(
    ws_url=settings.telemetry_ws,
    replay_path=settings.telemetry_replay,
    window_seconds=10.0,
    max_frames=int(settings.prediction_window_seconds * 2 * 10) + 500,
)


@dataclass
class AppState:
    settings: object
    telemetry: TelemetryIngest
    pipeline: AssistantPipeline
    evaluator: ProactiveEvaluator | None = None


app_state: AppState


async def _proactive_loop(evaluator: ProactiveEvaluator, poll_interval_s: float) -> None:
    while True:
        try:
            await evaluator.poll()
        except Exception as exc:  # noqa: BLE001 - proactive must never kill the API
            logger.warning("proactive poll failed", extra={"payload": {"error": str(exc)}})
        await asyncio.sleep(poll_interval_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    pipeline = AssistantPipeline(telemetry)
    restore_release(pipeline)

    evaluator: ProactiveEvaluator | None = None
    poll_task: asyncio.Task | None = None
    if settings.proactive_enabled:
        evaluator = ProactiveEvaluator(
            telemetry=telemetry,
            vehicle_id="ORION-X-TEST-01",
            poll_interval_s=settings.proactive_poll_interval_s,
            get_explanation=pipeline._fetch_prediction_explanation,
        )
        poll_task = asyncio.create_task(_proactive_loop(evaluator, settings.proactive_poll_interval_s))

    app_state = AppState(settings=settings, telemetry=telemetry, pipeline=pipeline, evaluator=evaluator)
    app.state.pipeline = pipeline
    app.state.telemetry = telemetry
    app.state.settings = settings
    app.state.evaluator = evaluator
    await telemetry.start()
    yield
    if poll_task is not None:
        poll_task.cancel()
    await telemetry.stop()
    await pipeline.aclose()


app = FastAPI(
    title="AURORA Assistant API",
    version=__version__,
    description="Grounds driver questions in live vehicle telemetry + a versioned RAG knowledge base.",
    lifespan=lifespan,
)

app.include_router(router, prefix="/v1")
app.include_router(operations_router, prefix="/v1/ops")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "assistant-api", "telemetry_source": telemetry.source}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "frames": len(telemetry.frames)}


@app.get("/metrics")
def metrics_endpoint():
    if not metrics_available():
        return Response("prometheus disabled", media_type="text/plain")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.mount("/", StaticFiles(directory="app/static", html=True), name="dashboard")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_config=None)