"""HTTP surface of the assistant API."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from vehicle_ai.schemas import AssistantRequest, AssistantResponse, ProactiveAskRequest

router = APIRouter()

_VOICE_CACHE = Path("/tmp/aurora-voice")


class EnvironmentInfo(BaseModel):
    service: str
    version: str
    prompt_version: str
    rag_index_version: str
    environment: str
    telemetry_source: str
    gateway_url: str
    rag_url: str


class TripRequest(BaseModel):
    """Relay of the simulator's destination-aware trip start."""

    destination_km: float = Field(gt=0)
    cabin_set_temp_c: float = 22.0
    ambient_temp_c: float = 15.0
    speed_profile_kph: list[float] = Field(default_factory=lambda: [60.0, 90.0, 110.0])
    segment_s: float = Field(default=120.0, ge=1.0)


class ScenarioRequest(BaseModel):
    type: str
    duration_s: float = 45.0
    intensity: float = 0.6


class SpeedControlRequest(BaseModel):
    speed_kph: float = Field(ge=0.0, le=200.0)


class HvacControlRequest(BaseModel):
    cabin_temp_c: float = Field(ge=12.0, le=30.0)


async def _relay_sim(request: Request, method: str, path: str, **kwargs) -> dict:
    """Thin command relay to the vehicle (simulator). Best-effort: an unreachable
    or rejecting vehicle surfaces as an explicit HTTP error, never a fake ack."""
    base = request.app.state.settings.telemetry_http
    client = request.app.state.pipeline._client
    try:
        resp = await client.request(method, f"{base}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) or 502
        raise HTTPException(status_code=status, detail=f"vehicle driver unavailable: {exc}") from exc


@router.post("/ask", response_model=AssistantResponse)
async def ask(req: AssistantRequest, request: Request):
    try:
        return await request.app.state.pipeline.ask(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"assistant pipeline unavailable: {exc}") from exc


@router.get("/wire/live")
def live():
    return setup().live_summary()


@router.get("/proactive")
def proactive(request: Request):
    """Deterministic alert snapshot — code-only, no LLM (WS4 option a)."""
    evaluator = getattr(request.app.state, "evaluator", None)
    if evaluator is None:
        raise HTTPException(status_code=503, detail="proactive engine unavailable")
    return evaluator.snapshot()


@router.post("/proactive/ask", response_model=AssistantResponse)
async def proactive_ask(req: ProactiveAskRequest, request: Request):
    """Word the current top alert(s) through the assistant (WS4 option b).

    The proactive engine has already decided the events; the pipeline only
    styles them. ``ack_event_ids`` dismisses alerts so they are not re-surfaced
    until the event resolves and recurs.
    """
    evaluator = getattr(request.app.state, "evaluator", None)
    if evaluator is None:
        raise HTTPException(status_code=503, detail="proactive engine unavailable")
    evaluator.ack(req.ack_event_ids)
    events = evaluator.top_active(req.count or 1)
    try:
        return await request.app.state.pipeline.word_proactive(
            events, session_id=req.session_id, vehicle_id=req.vehicle_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"assistant pipeline unavailable: {exc}") from exc


@router.get("/environment")
def environment():
    from vehicle_ai.version import RAG_INDEX_VERSION, TELEMETRY_SCHEMA_VERSION, __version__

    c = setup()
    return EnvironmentInfo(
        service="assistant-api",
        version=__version__,
        prompt_version=c.settings.prompt_version,
        rag_index_version=RAG_INDEX_VERSION,
        environment=c.settings.environment,
        telemetry_source=c.telemetry.source,
        gateway_url=c.settings.gateway_url,
        rag_url=c.settings.rag_url,
    )


@router.get("/hud")
async def hud(request: Request, prediction: str = "0"):
    """WS5 — thin facade over frozen system state (provenance-labelled).

    Aggregates values their owners already decided: live telemetry buffer,
    WS4 proactive engine, prediction service status, gateway provider table,
    RAG index metadata, service readiness. ``?prediction=1`` additionally
    fetches the deterministic explanation of the current SOC-drop prediction.
    """
    include_explanation = prediction.strip() in {"1", "true", "yes"}
    evaluator = getattr(request.app.state, "evaluator", None)
    from app.api.hud import collect_hud

    return await collect_hud(request.app.state.pipeline, evaluator, include_explanation=include_explanation)


@router.get("/latest")
def latest(request: Request):
    """Last raw live frame the assistant is grounding on (10 Hz).

    Feeds the immersive cockpit's road scene so the world moves exactly as far
    as the vehicle — telemetry is the source of truth, never an integrated guess.
    """
    frames = request.app.state.telemetry.frames
    if not frames:
        return {"status": "no-telemetry"}
    return frames[-1]


# -- live driver command surfaces (relayed to the vehicle) -------------------


@router.get("/control")
async def control(request: Request):
    """Current commanded state of driver-controlled surfaces."""
    return await _relay_sim(request, "GET", "/api/control")


@router.post("/control/speed")
async def control_speed(req: SpeedControlRequest, request: Request):
    return await _relay_sim(request, "POST", "/api/control/speed", json=req.model_dump())


@router.post("/control/hvac")
async def control_hvac(req: HvacControlRequest, request: Request):
    return await _relay_sim(request, "POST", "/api/control/hvac", json=req.model_dump())


@router.post("/control/trip")
async def control_trip(req: TripRequest, request: Request):
    return await _relay_sim(request, "POST", "/api/trip", json=req.model_dump())


@router.post("/control/scenario")
async def control_scenario(req: ScenarioRequest, request: Request):
    return await _relay_sim(request, "POST", "/api/scenario", json=req.model_dump())


# ---------------------------------------------------------------------------
# WS6 — Scenario Director: one-click composite launches, all through the real
# simulator (never faked). Each scenario is an ordered list of genuine vehicle
# commands; the caller watches telemetry/prediction/alerts react to them.
# ---------------------------------------------------------------------------

_SCENARIO_DIRECTOR: list[dict] = [
    {
        "key": "normal",
        "title": "Normal drive",
        "tagline": "Clear faults, free driving, nominal climate.",
        "steps": [("POST", "/api/scenario", {"type": "none", "duration_s": 1, "intensity": 0})],
    },
    {
        "key": "highway_cold",
        "title": "Highway + cold",
        "tagline": "Speed 120 · outside −5 °C · cabin 24 °C.",
        "steps": [
            ("POST", "/api/scenario", {"type": "none", "duration_s": 1, "intensity": 0}),
            ("POST", "/api/trip", {"destination_km": 120, "cabin_set_temp_c": 24, "ambient_temp_c": -5, "speed_profile_kph": [115, 130], "segment_s": 120}),
            ("POST", "/api/control/speed", {"speed_kph": 120}),
        ],
    },
    {
        "key": "low_arrival",
        "title": "Low arrival energy",
        "tagline": "Long destination · pre-drained pack (SOC~15%).",
        "steps": [
            ("POST", "/api/scenario", {"type": "low_energy", "duration_s": 1, "intensity": 1.0}),
            ("POST", "/api/trip", {"destination_km": 220, "cabin_set_temp_c": 20, "ambient_temp_c": 4, "speed_profile_kph": [100, 115], "segment_s": 120}),
        ],
    },
    {
        "key": "tire_leak",
        "title": "Tyre leak",
        "tagline": "RR pressure starts dropping while driving at 80 km/h.",
        "steps": [
            ("POST", "/api/control/speed", {"speed_kph": 80}),
            ("POST", "/api/scenario", {"type": "leak", "duration_s": 300, "intensity": 1.0}),
        ],
    },
    {
        "key": "charging_stall",
        "title": "Charging stall",
        "tagline": "Plugged in, but charging power stays at 0 kW.",
        "steps": [
            ("POST", "/api/scenario", {"type": "charge_stall", "duration_s": 300, "intensity": 1.0}),
        ],
    },
    {
        "key": "battery_heat",
        "title": "Battery heat",
        "tagline": "Cooling degraded · pack temperature rises while driving.",
        "steps": [
            ("POST", "/api/control/speed", {"speed_kph": 110}),
            ("POST", "/api/scenario", {"type": "battery_heat", "duration_s": 360, "intensity": 1.0}),
        ],
    },
]

SCENARIO_BY_KEY = {s["key"]: s for s in _SCENARIO_DIRECTOR}


class ScenarioLaunch(BaseModel):
    key: str


@router.get("/scenarios")
def scenarios():
    """Catalog for the cockpit's Scenario Director (backend-driven, real steps)."""
    return [{"key": s["key"], "title": s["title"], "tagline": s["tagline"], "step_count": len(s["steps"])} for s in _SCENARIO_DIRECTOR]


@router.post("/scenarios/{key}")
async def launch_scenario(key: str, request: Request):
    """Launch a composite scenario through the real vehicle relay.

    Executes the vehicle commands in order; the state you see afterwards is the
    simulator's own reaction — nothing is returned that the sim did not do.
    """
    spec = SCENARIO_BY_KEY.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {key}")
    executed: list[dict] = []
    for method, path, body in spec["steps"]:
        try:
            result = await _relay_sim(request, method, path, json=body)
            executed.append({"method": method, "path": path, "body": body, "result": result})
        except HTTPException as exc:
            control = None
            try:
                control = await _relay_sim(request, "GET", "/api/control")
            except HTTPException:
                pass
            return {
                "key": key, "title": spec["title"], "complete": False,
                "error": exc.detail, "steps": executed, "control": control,
            }
    control = await _relay_sim(request, "GET", "/api/control")
    return {"key": key, "title": spec["title"], "complete": True, "steps": executed, "control": control}


# ---------------------------------------------------------------------------
# WS6 — Why? / What-if? hero interactions. Deterministic code answers the
# numbers; the frontend speaks/narrates them. No LLM invention.
# ---------------------------------------------------------------------------


def _why_narrative(exp: dict) -> str:
    pred = exp.get("to_prediction") or {}
    causes = exp.get("top_causes") or []
    drop = pred.get("calibrated_soc_drop_pct")
    lines = []
    if exp.get("status") != "active":
        lines.append("The calibrated model is currently unavailable, so this uses the physics estimate only.")
    if drop is not None:
        lines.append(f"Projected battery discharge is {drop:.2f} SOC points per five minutes.")
    ranked = sorted(causes, key=lambda c: abs(c.get("contribution_soc_pct", 0.0) or 0.0), reverse=True)
    if ranked:
        leader = ranked[0]
        contribution = leader.get("contribution_soc_pct") or 0.0
        verb = "raises" if contribution >= 0 else "lowers"
        lines.append(
            f"The largest driver is {leader.get('label') or leader.get('key')}, which {verb} the projected drop by {abs(contribution):.2f} points."
        )
        for c in ranked[1:4]:
            contribution = c.get("contribution_soc_pct") or 0.0
            verb = "raises" if contribution >= 0 else "lowers"
            lines.append(f"Next, {c.get('label') or c.get('key')} {verb} it by {abs(contribution):.2f} points.")
    lines.append("These numbers are computed from the live window by the calibration stack.")
    return " ".join(lines)


@router.post("/explain/why")
async def explain_why(request: Request):
    """Why is the prediction what it is? Forwards the live window to the
    deterministic explainer and styles the ranked causes in code."""
    pipeline = request.app.state.pipeline
    try:
        explanation = await pipeline._fetch_prediction_explanation()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"explain unavailable: {exc}") from exc
    if explanation is None:
        return {"kind": "why", "explanation": None, "narrative": "Not enough telemetry yet to build a prediction window — keep driving and ask again.", "status": "collecting"}
    exp = explanation.model_dump(mode="json")
    return {"kind": "why", "explanation": exp, "narrative": _why_narrative(exp), "status": exp.get("status")}


class WhatIfRequest(BaseModel):
    """Proposed change over the live condition: omit a field to leave it as-is."""

    speed_kph: float | None = Field(default=None, ge=0.0, le=200.0, description="proposed cruise speed (km/h)")
    cabin_temp_c: float | None = Field(default=None, ge=12.0, le=30.0, description="proposed cabin set point (°C)")


@router.post("/explain/whatif")
async def explain_whatif(req: WhatIfRequest, request: Request):
    """Directional physics recalc: what happens to arrival SOC if the driver
    changes speed / cabin? Uses the same energy model as the simulator."""
    frames = request.app.state.telemetry.frames
    if not frames:
        raise HTTPException(status_code=503, detail="no telemetry yet — start driving first")
    sig = frames[-1].get("signals", {})
    soc = sig.get("battery_soc_pct")
    speed = sig.get("vehicle_speed_kph")
    outside = sig.get("outside_temp_c") or sig.get("ambient_temp_c")
    if soc is None or speed is None:
        raise HTTPException(status_code=503, detail="required signals missing from the live frame")

    control = await _relay_sim(request, "GET", "/api/control")
    setpoint = control.get("cabin_set_temp_c") or 18.0
    ambient = control.get("ambient_temp_c") if control.get("ambient_temp_c") is not None else (outside or 15.0)
    remaining = control.get("remaining_km")
    if remaining is None and control.get("trip_active"):
        trip = await _relay_sim(request, "GET", "/api/trip")
        remaining = (trip.get("trip") or {}).get("remaining_km")
    distance = remaining if remaining and remaining > 0 else 100.0

    delta_speed = (req.speed_kph - speed) if req.speed_kph is not None else 0.0
    delta_cabin = (req.cabin_temp_c - setpoint) if req.cabin_temp_c is not None else 0.0
    body = {
        "distance_km": round(distance, 1),
        "battery_wh": 82_000.0,  # simulator usable-capacity profile (simparams)
        "soc_pct": soc,
        "speed_kph": speed,
        "cabin_temp_c": (req.cabin_temp_c if req.cabin_temp_c is not None else setpoint),
        "ambient_temp_c": ambient,
        "delta_speed_kph": round(delta_speed, 1),
        "delta_cabin_temp_c": round(delta_cabin, 1),
    }
    result = await _relay_sim(request, "POST", "/api/trip/what-if", json=body)
    base = result.get("baseline_arrival_soc_pct")
    after = result.get("perturbed_arrival_soc_pct")
    delta = result.get("delta_arrival_soc_pct")
    parts = []
    cause = []
    if req.speed_kph is not None:
        parts.append(f"at {req.speed_kph:.0f} km/h")
        cause.append(f"{'slowing' if delta_speed < 0 else 'speeding'} to {req.speed_kph:.0f}")
    if req.cabin_temp_c is not None:
        parts.append(f"cabin {req.cabin_temp_c:.0f} °C")
        cause.append(f"climate at {req.cabin_temp_c:.0f} °C")
    scenario = " and ".join(cause) or "the proposed change"
    narrative = (
        f"Projected arrival SOC over that {distance:.0f} km leg is {base:.1f}% "
        f"as-is; {scenario}, it would end at about {after:.1f}% "
        f"({'+' if (delta or 0) >= 0 else ''}{delta:.1f} points). This is a directional physics estimate over the live window."
    )
    return {"kind": "whatif", "request": body, "result": result, "narrative": narrative}


# ---------------------------------------------------------------------------
# WS6 — Voice studio. macOS ``say`` neural voices served as real audio, with
# the browser speechSynthesis as the documented fallback. AURORA only speaks
# grounded answers / code-decided alerts regardless of the engine.
# ---------------------------------------------------------------------------


def _say_available() -> bool:
    return sys.platform == "darwin" and shutil.which("say") is not None


def _say_voices() -> list[dict]:
    import re  # noqa: PLC0415 - local: keeps module import independent of macOS say

    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    voices: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            name, rest = parts[0].strip(), parts[1].split("#", 1)[0].strip()
        else:
            # macOS variants that space-pad columns:  Name  <langs>  # sample
            m = re.match(r"^(?P<name>.+?)\s{2,}(?P<lang>[A-Za-z_\-]{2,8})\s{2,}(?:#\s*.*)?$", line)
            if not m:
                continue
            name, rest = m.group("name").strip(), m.group("lang").strip()
        if name:
            voices.append({"name": name, "lang": rest})
    return voices


_PREFERRED_SAY = ("Premium", "Enhanced", "Daniel", "Samantha", "Ava", "Allison", "Siri", "Fred")


def _natural_say_voice(voices: list[dict]) -> str | None:
    for preferred in _PREFERRED_SAY:
        for v in voices:
            if preferred.lower() in v["name"].lower() and v["lang"].split("_")[0] == "en":
                return v["name"]
    return voices[0]["name"] if voices else None


@router.get("/voice")
def voice_status():
    voices = _say_voices() if _say_available() else []
    return {
        "engine": "studio-say" if _say_available() else "browser",
        "available": _say_available(),
        "default_voice": _natural_say_voice(voices),
        "voices": voices,
    }


@router.get("/voice/voices")
def voice_list():
    return _say_voices() if _say_available() else []


@router.get("/voice/say")
def voice_say(text: str, voice: str | None = None, rate: float = 170.0, pitch: float = 1.0):
    """Synthesize text with the macOS neural ``say`` engine (real audio, cached).

    Falls back to ``414``-free honest 404 when the host cannot synthesize.
    """
    if not text or not _say_available():
        raise HTTPException(status_code=404, detail="voice engine unavailable on this host")
    text = text[:400]
    voices = _say_voices()
    chosen = voice and any(v["name"] == voice for v in voices)
    voice_name = voice if chosen else (_natural_say_voice(voices) or "")
    _VOICE_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{text}|{voice_name}|{rate}|{pitch}".encode("utf-8")).hexdigest()[:20]
    wav = _VOICE_CACHE / f"{key}.wav"
    if not wav.exists():
        cmd = ["say"]
        if voice_name:
            cmd += ["-v", voice_name]
        cmd += ["-r", str(int(rate)), "-o", str(wav), "--file-format=WAVE", "--data-format=LEI16@22050", text]
        try:
            subprocess.run(cmd, capture_output=True, timeout=40, check=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"say failed: {exc}") from exc
    return Response(content=wav.read_bytes(), media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/healthz")
def healthz():
    return {"status": "ok", "service": "assistant-api", "telemetry_source": setup().telemetry.source}


@router.get("/readyz")
def readyz():
    return {"status": "ok", "frames": len(setup().telemetry.frames)}


def setup():
    from app.main import app_state

    return app_state

@router.post("/control/demo/reset")
async def reset_demo(request: Request):
    return await _relay_sim(request, "POST", "/api/demo/reset", json={})
