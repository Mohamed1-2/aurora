"""Telemetry simulator service."""

from __future__ import annotations

import os
import json
import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from vehicle_ai.logging_conf import configure_logging, get_logger
from vehicle_ai.schemas import (
    ArrivalEstimate,
    CalibratedSoCDropRequest,
    CalibratedSoCDropResponse,
    DTC,
    EnergyFeatures,
    ExplainFromTelemetryRequest,
    ExplainSoCDropRequest,
    GPSPosition,
    PredictionExplanation,
    SimulatorTripRequest,
    TelemetryFrame,
    TelemetryWindow,
    TripContext,
    TUMReplayRequest,
    TUMReplaySlice,
    WarningMessage,
    WhatIfRequest,
    WhatIfResponse,
)
from vehicle_ai.telemetry_utils import aggregate_window
from vehicle_ai.version import TELEMETRY_SCHEMA_VERSION, __version__

from app.config import settings
from app.data.dtc_catalog import DTC_CATALOG, SIGNAL_CATALOG
from app.engine.calibrated_soc_drop import calibrated_service
from app.engine.energy_engine import compute_drivers, project_arrival, what_if
from app.engine.explain_from_telemetry import explain_from_frames
from app.engine.scenarios import SCENARIOS
from app.engine.tum_replay import DataUnavailableError, TUMReplaySource
from app.engine.vehicle_model import VehicleModel

configure_logging("telemetry-simulator", settings.log_level)
logger = get_logger("aurora.telemetry")


class ScenarioRequest(BaseModel):
    type: str
    duration_s: float = 45.0
    intensity: float = 0.6


class SpeedControlRequest(BaseModel):
    speed_kph: float = Field(ge=0.0, le=200.0)


class HvacControlRequest(BaseModel):
    cabin_temp_c: float = Field(ge=12.0, le=30.0)


class Simulator:
    """Owns the vehicle model + ring buffer + broadcast hub."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.model = VehicleModel(settings.vehicle_id, seed=settings.seed)
        self.dt = 1.0 / settings.hz
        self.buffer: deque[dict] = deque(maxlen=settings.buffer_frames)
        self.subscribers: set[asyncio.Queue] = set()
        self.start_ts = time.time()
        self.last_frame: dict | None = None

    def frame_to_schema(self, raw: dict) -> TelemetryFrame:
        dtcs = [
            DTC(code=c, description=DTC_CATALOG[c]["description"], system=DTC_CATALOG[c]["system"], severity=DTC_CATALOG[c]["severity"])
            for c in raw["active_dtcs"]
            if c in DTC_CATALOG
        ]
        warnings = [WarningMessage(**w) for w in raw["warnings"]]
        return TelemetryFrame(
            vehicle_id=raw["vehicle_id"],
            timestamp=raw["timestamp"],
            seq=raw["seq"],
            driving_mode=raw["driving_mode"],
            charging=raw["charging"],
            signals=raw["signals"],
            active_dtcs=dtcs,
            warnings=warnings,
            gps=GPSPosition(**raw["gps"]),
        )

    def advance(self) -> TelemetryFrame:
        raw = self.model.tick(self.dt)
        frame = self.frame_to_schema(raw)
        self.buffer.append(frame.model_dump())
        self.last_frame = frame
        return frame

    def window(self, seconds: float = 15.0) -> TelemetryWindow:
        frames = list(self.buffer)
        n = max(1, int(seconds * self.settings.hz))
        sample = frames[-n:] if len(frames) >= n else frames
        agg = aggregate_window(sample, seconds)
        return TelemetryWindow(
            vehicle_id=settings.vehicle_id,
            frames=agg["frames"],
            window_seconds=agg["window_seconds"],
            start_ts=agg["start_ts"],
            end_ts=agg["end_ts"],
            mean=agg["mean"],
            min=agg["min"],
            max=agg["max"],
            std=agg["std"],
        )


sim = Simulator(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(emit_loop())
    yield
    task.cancel()


async def emit_loop() -> None:
    while True:
        frame = sim.advance()
        stale: list[asyncio.Queue] = []
        for q in sim.subscribers:
            try:
                q.put_nowait(frame.model_dump())
            except asyncio.QueueFull:
                stale.append(q)
        for q in stale:
            sim.subscribers.discard(q)
        await asyncio.sleep(sim.dt)


app = FastAPI(
    title="AURORA Telemetry Simulator",
    version=__version__,
    description="Synthetic (physics-lite) vehicle data stream for the AURORA platform.",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "telemetry-simulator", "version": __version__, "scenario": sim.model.scenario.label, "mode": sim.model.scenario.mode}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "frames": len(sim.buffer), "uptime_s": round(time.time() - sim.start_ts, 1)}


@app.get("/schema-version")
def schema_version():
    return {"telemetry_schema_version": TELEMETRY_SCHEMA_VERSION}


@app.get("/latest", response_model=TelemetryFrame)
def latest():
    if sim.last_frame is None:
        return sim.advance()
    return sim.last_frame


@app.get("/window", response_model=TelemetryWindow)
def window(seconds: float = 15.0):
    return sim.window(seconds)


@app.get("/signals")
def signals():
    return {"signal_catalog": {k: {"unit": u, "description": d, "min": lo, "max": hi} for k, (u, d, lo, hi) in SIGNAL_CATALOG.items()}}


@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest):
    if req.type not in SCENARIOS:
        return {"error": f"unknown scenario, choose from {SCENARIOS}"}
    sim.model.configure(req.type, req.duration_s, req.intensity)
    logger.info(f"scenario activated", extra={"payload": {"scenario": req.type, "duration_s": req.duration_s, "intensity": req.intensity}})
    return {"scenario": req.type, "duration_s": req.duration_s, "active": True}


# ---------------------------------------------------------------------------
# Live driver command surfaces (vehicle-side). The assistant relays these to the
# driver UI; they operate the vehicle model under test and feed the SAME frames
# the frozen intelligence (proactive, prediction, RAG, LLM) reasons over.
# ---------------------------------------------------------------------------


@app.get("/api/control")
def control_status():
    """Current commanded state of driver-controlled surfaces (not telemetry)."""
    return sim.model.control_status()


@app.post("/api/control/speed")
def control_speed(req: SpeedControlRequest):
    """Live cruise command (kph). Follow the active plan again by omitting."""
    value = sim.model.set_target_speed(req.speed_kph)
    logger.info("speed command", extra={"payload": {"speed_kph": req.speed_kph, "accepted": value}})
    return {"target_speed_kph": value, "accepted": True}


@app.post("/api/control/hvac")
def control_hvac(req: HvacControlRequest):
    """Update the cabin set point of the active trip (HVAC exists only in trip)."""
    accepted = sim.model.set_cabin_set_temp(req.cabin_temp_c)
    if accepted is None:
        raise HTTPException(status_code=409, detail="no active trip — start a destination to enable climate")
    logger.info("hvac command", extra={"payload": {"cabin_set_temp_c": req.cabin_temp_c, "accepted": accepted}})
    return {"cabin_set_temp_c": accepted, "hvac_power_kw": sim.model.control_status()["hvac_power_kw"]}


# ---------------------------------------------------------------------------
# AURORA v2 WS1 — Trip & Energy (scenario simulator) + genuine TUM replay.
# Two explicit, non-blended modes:
#   mode=scenario    -> synthetic/controlled causal trip
#   mode=tum_replay  -> genuine measured TUM telemetry from raw parquet
# ---------------------------------------------------------------------------

replay_source = TUMReplaySource(settings.tum_uds_dir or None)


@app.post("/api/trip", response_model=dict)
def start_trip(req: SimulatorTripRequest):
    """Start a destination-aware synthetic trip (SCENARIO mode)."""
    sim.model.configure_trip(
        destination_km=req.destination_km,
        cabin_set_temp_c=req.cabin_set_temp_c,
        ambient_temp_c=req.ambient_temp_c,
        speed_profile_kph=req.speed_profile_kph,
        segment_s=req.segment_s,
    )
    logger.info(f"trip started", extra={"payload": req.model_dump()})
    trip = sim.model.scenario.trip
    return {
        "mode": "scenario",
        "active": True,
        "trip": {
            "destination_km": trip.destination_km,
            "cabin_set_temp_c": trip.cabin_set_temp_c,
            "ambient_temp_c": trip.ambient_temp_c,
            "speed_profile_kph": trip.speed_profile_kph,
        },
    }


@app.get("/api/trip", response_model=dict)
def trip_status():
    trip = sim.model.scenario.trip
    if trip is None:
        return {"mode": sim.model.scenario.mode, "active": False, "trip": None}
    return {
        "mode": "scenario",
        "active": True,
        "trip": {
            "destination_km": trip.destination_km,
            "remaining_km": round(trip.remaining_km, 2),
            "cabin_set_temp_c": trip.cabin_set_temp_c,
            "ambient_temp_c": trip.ambient_temp_c,
            "completed": trip.completed,
        },
    }


@app.post("/api/trip/energy-drivers", response_model=EnergyFeatures)
def energy_drivers(speed_kph: float, cabin_temp_c: float | None = None, ambient_temp_c: float | None = None):
    """Instantaneous battery power demand split by driver (physics baseline)."""
    return EnergyFeatures(**compute_drivers(speed_kph, cabin_temp_c=cabin_temp_c, ambient_temp_c=ambient_temp_c).to_dict())


@app.post("/api/trip/arrival-estimate", response_model=ArrivalEstimate)
def arrival_estimate(distance_km: float, speed_kph: float, cabin_temp_c: float | None = None, ambient_temp_c: float | None = None):
    """Physics projection of arrival SOC (consistency, NOT real-EV accuracy)."""
    from app.engine.simparams import DEFAULT_PROFILE as _P

    est = project_arrival(
        distance_km=distance_km,
        battery_wh=_P.value("battery_usable_wh"),
        soc_pct=sim.model.battery_soc,
        speed_kph=speed_kph,
        cabin_temp_c=cabin_temp_c,
        ambient_temp_c=ambient_temp_c,
    )
    return ArrivalEstimate(**est.to_dict())


@app.post("/api/trip/what-if", response_model=WhatIfResponse)
def trip_what_if(req: WhatIfRequest):
    """Directional what-if (monotonicity testable; not an accuracy claim)."""
    res = what_if(
        distance_km=req.distance_km,
        battery_wh=req.battery_wh,
        soc_pct=req.soc_pct,
        speed_kph=req.speed_kph,
        delta_speed_kph=req.delta_speed_kph,
        delta_cabin_temp_c=req.delta_cabin_temp_c,
        ambient_temp_c=req.ambient_temp_c,
    )
    return WhatIfResponse(**res)


@app.get("/api/tum/status")
def tum_status():
    return replay_source.describe()


@app.post("/api/trip/replay-tum")
def replay_tum(req: TUMReplayRequest):
    """Replay a genuine measured TUM slice from raw parquet (mode=tum_replay).

    Never fabricates: raw parquet missing => clean 503 (DataUnavailableError).
    """
    try:
        frames = replay_source.slice(
            vehicle_id=req.vehicle_id,
            episode_id=req.episode_id,
            start_index=req.start_index,
            duration_windows=req.duration_windows,
        )
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    sim.model.scenario.set_replay_mode()
    return {
        "mode": "tum_replay",
        "vehicle_id": req.vehicle_id,
        "episode_id": frames[0]["episode_id"] if frames else None,
        "frames": len(frames),
        "measured_signals": sorted({k for f in frames for k in f["signals"]}),
        "slices": frames,
    }


# ---------------------------------------------------------------------------
# WS2 — calibrated SOC-drop prediction (active path with physics fallback)
# ---------------------------------------------------------------------------


@app.get("/api/prediction/status")
def prediction_status():
    return calibrated_service.status()


@app.post("/api/prediction/calibrated-soc-drop", response_model=CalibratedSoCDropResponse)
def calibrated_soc_drop(req: CalibratedSoCDropRequest):
    """Calibrated SOC-drop for a 300s window (frozen v2-ws1-r1).

    Active path: physics + lam_band * ml_residual. Returns
    ``physics_only_fallback`` (never raises) when the ML artifact is unavailable.
    """
    return calibrated_service.predict(req)


@app.post("/api/prediction/explain-soc-drop", response_model=PredictionExplanation)
def explain_soc_drop(req: ExplainSoCDropRequest):
    """Deterministic explanation of a prediction (WS3).

    ``{"to": {features}}`` -> absolute explanation ("why is the prediction what
    it is"); ``{"from": {...}, "to": {...}}`` -> delta explanation ("why did it
    change"). Every number / ranked cause is computed in code (physics drivers +
    SHAP on the frozen model); the LLM only styles this block.
    """
    return calibrated_service.explain(req)


@app.post("/api/prediction/explain-from-telemetry", response_model=PredictionExplanation)
def explain_from_telemetry(req: ExplainFromTelemetryRequest):
    """Explain a prediction from raw telemetry frames the assistant already holds.

    The assistant forwards the live frames it grounded on (``to_frames``, plus
    ``from_frames`` for a delta/what-if), and this service bins them and reuses
    the exact frozen feature extraction + physics baseline. Never raises: with
    too few frames it returns a physics-only ``PredictionExplanation``.
    """
    return explain_from_frames(
        to_frames=req.to_frames,
        from_frames=req.from_frames,
        hz=req.hz,
        capacity_kwh=req.capacity_kwh,
    )


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    sim.subscribers.add(q)
    try:
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(frame))
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        sim.subscribers.discard(q)


@app.get("/metrics")
async def metrics():
    from vehicle_ai.metrics import metrics_available
    if not metrics_available():
        return {"prometheus": "unavailable"}
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from vehicle_ai.metrics import MetricsRegistry
    reg = MetricsRegistry()
    frame = sim.last_frame
    if frame:
        for k, v in frame.signals.items():
            if v == v:  # skip NaN
                reg.signal_gauge.labels(frame.vehicle_id, k).set(v)
    from starlette.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_config=None)

@app.post("/api/demo/reset")
async def reset_demo():
    """Explicit demo reset: replace simulator state, never physical controls."""
    seq = sim.model.seq
    model = VehicleModel(settings.vehicle_id, seed=42)
    model.seq = seq
    model.battery_soc = 65.0
    model.configure_trip(8.0, 22.0, 18.0, speed_profile_kph=[30, 50, 65, 40], segment_s=45.0)
    sim.model = model
    sim.buffer.clear()
    sim.last_frame = None
    return {"status":"started", "source":"simulator", "destination_km":8, "battery_soc_pct":65}
