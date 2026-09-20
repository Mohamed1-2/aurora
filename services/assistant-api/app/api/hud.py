"""``GET /v1/hud`` — thin read-only facade aggregating frozen system state.

The facade collects numbers that are each *owned elsewhere* and already
decided: the assistant's live telemetry buffer, the WS4 proactive engine, the
prediction service status, the gateway provider table, the RAG index metadata,
and each upstream service's readiness. It performs **no inference of its own** —
whatever is computed (proactive triggers, aggregation, coverage) was computed by
the owning service and is only re-labelled here.

Provenance is a first-class field: every displayed value declares where it came
from. The taxonomy (see ``PROVENANCE_LEGEND``) is:

* ``measured``     — raw quantitative telemetry from the vehicle/simulator;
* ``prediction``   — output of the calibrated SOC-drop model (physics + ML);
* ``derived``      — computed in code on top of other values (proactive
                     triggers/evidence, aggregation, service health);
* ``documentation``— text/metadata from the versioned RAG knowledge base;
* ``simulator``    — telemetry that is *simulated* (this repo runs a simulator
                     as the vehicle; production streaming would be ``measured``).

On the local/demo stack the assistant is wired to the simulator, so its signals
carry ``provenance="simulator"`` and the block states so explicitly. A real
vehicle deployment (``AURORA_ASSISTANT_ENVIRONMENT=production``) yields
``measured``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from vehicle_ai.version import PROMPT_VERSION, RAG_INDEX_VERSION, __version__

PROVENANCE_LEGEND: dict[str, str] = {
    "measured": "Raw quantitative value from the vehicle's telemetry stream.",
    "prediction": "Output of the calibrated SOC-drop model (physics baseline + calibrated ML residual).",
    "derived": "Computed in code from the values above (proactive triggers, aggregation, service health).",
    "documentation": "Text or metadata from the versioned RAG knowledge base.",
    "simulator": "Telemetry synthesized by the simulator under test on this stack — not a real vehicle.",
}

# (signal, unit, label, nominal_min, nominal_max). Nominal bounds are only for
# the glance gauges; they are not sensor limits.
SIGNAL_SPECS: list[tuple[str, str, str, float | None, float | None]] = [
    ("battery_soc_pct", "%", "Battery SOC", 0.0, 100.0),
    ("distance_to_empty_km", "km", "Remaining range", 0.0, 600.0),
    ("vehicle_speed_kph", "km/h", "Speed", 0.0, 200.0),
    ("motor_power_kw", "kW", "Motor power", -180.0, 180.0),
    ("charge_rate_kw", "kW", "Charge rate", 0.0, 150.0),
    ("battery_temp_c", "degC", "Battery pack temperature", -40.0, 120.0),
    ("coolant_temp_c", "degC", "Coolant temperature", -40.0, 120.0),
    ("tire_pressure_fl_kpa", "kPa", "Tyre pressure · front left", 150.0, 350.0),
    ("tire_pressure_fr_kpa", "kPa", "Tyre pressure · front right", 150.0, 350.0),
    ("tire_pressure_rl_kpa", "kPa", "Tyre pressure · rear left", 150.0, 350.0),
    ("tire_pressure_rr_kpa", "kPa", "Tyre pressure · rear right", 150.0, 350.0),
    ("pack_current_a", "A", "Pack current", -400.0, 400.0),
    ("battery_voltage_v", "V", "Pack voltage", 0.0, 900.0),
    ("odometer_km", "km", "Odometer", 0.0, None),
    ("cabin_set_temp_c", "degC", "Cabin set point", 12.0, 30.0),
    ("cabin_temp_c", "degC", "Cabin temperature", 12.0, 40.0),
    ("outside_temp_c", "degC", "Outside air temperature", -30.0, 50.0),
    ("ambient_temp_c", "degC", "Ambient temperature", -30.0, 50.0),
    ("destination_remaining_km", "km", "Distance to destination", 0.0, 600.0),
    ("arrival_eta_min", "min", "Arrival ETA", 0.0, 600.0),
    ("hvac_power_kw", "kW", "Climate power", 0.0, 12.0),
    ("trip_distance_km", "km", "Trip distance", 0.0, None),
]

# Signals that get a 60 s sparkline on the HUD.
SPARK_SIGNALS = [
    "vehicle_speed_kph",
    "battery_soc_pct",
    "battery_temp_c",
    "motor_power_kw",
    "charge_rate_kw",
    "tire_pressure_fl_kpa",
    "tire_pressure_fr_kpa",
    "tire_pressure_rl_kpa",
    "tire_pressure_rr_kpa",
]

_NOMINAL_FULL_COVERAGE_FRAMES = 3000  # 300 s @ 10 Hz


def signals_metadata(
    grounding: dict[str, Any],
    simulated: bool,
    spark_getter: Callable[[str], list[list[float]]],
) -> dict[str, dict[str, Any]]:
    """Label each signal's value with unit, nominal range, window stats and provenance.

    A signal that is absent from the live frame is reported as ``present=false``
    with ``value=None`` — silence is honest, never a zero.
    """
    raw = grounding.get("signals", {}) or {}
    mean = grounding.get("mean", {}) or {}
    mini = grounding.get("min", {}) or {}
    maxi = grounding.get("max", {}) or {}
    trending = grounding.get("trending", {}) or {}
    prov = "simulator" if simulated else "measured"
    out: dict[str, dict[str, Any]] = {}
    for sig, unit, label, lo, hi in SIGNAL_SPECS:
        present = sig in raw and raw[sig] is not None
        out[sig] = {
            "label": label,
            "unit": unit,
            "min": lo,
            "max": hi,
            "present": present,
            "value": raw.get(sig) if present else None,
            "window_min": mini.get(sig),
            "window_max": maxi.get(sig),
            "trending_per_min": trending.get(sig),
            "provenance": prov,
            "spark": spark_getter(sig) if (present and sig in SPARK_SIGNALS) else None,
        }
    return out


def telemetry_block(telemetry: Any, simulated: bool) -> dict[str, Any]:
    """Assemble the live telemetry block. No inference — just labelling."""
    grounding = telemetry.grounding()
    if not grounding:
        return {
            "provenance": "simulator" if simulated else "measured",
            "status": "no-telemetry",
            "frame_count": 0,
            "signals": {},
            "dtc_codes": [],
            "warnings": [],
        }
    from app.telemetry.ingest import TelemetryIngest

    spark_getter = (
        (lambda sig: [[round(float(ts), 2), round(float(v), 3)] for ts, v in telemetry.signal_series(sig, max_age_s=60.0, stride_s=1.0)])
        if isinstance(telemetry, TelemetryIngest)
        else (lambda sig: [])
    )
    return {
        "provenance": "simulator" if simulated else "measured",
        "status": "live",
        "source": grounding.get("source", "unknown"),
        "frame_count": len(telemetry.frames),
        "window_seconds": telemetry.window_seconds,
        "vehicle_id": grounding.get("vehicle_id"),
        "mode": grounding.get("mode"),
        "charging": bool(grounding.get("charging", False)),
        "signals": signals_metadata(grounding, simulated, spark_getter),
        "dtc_codes": [d.get("code") for d in grounding.get("dtcs", [])],
        "warnings": grounding.get("warnings", []),
    }


def proactive_block(evaluator: Any) -> dict[str, Any] | None:
    """WS4 alert snapshot — the engine already decided these events."""
    if evaluator is None:
        return None
    snap = evaluator.snapshot()
    return {
        "provenance": "derived",
        "generated_at": getattr(snap, "generated_at", None),
        "poll_interval_s": getattr(snap, "poll_interval_s", None),
        "active": [e.model_dump() if hasattr(e, "model_dump") else e for e in snap.active],
        "recent": [e.model_dump() if hasattr(e, "model_dump") else e for e in getattr(snap, "recent", [])],
        "detectors_eligible": list(snap.detectors_eligible),
        "detectors_skipped_no_signal": dict(snap.detectors_skipped_no_signal),
    }


def prediction_block(
    frame_count: int,
    service_status: dict[str, Any] | None,
    explanation: dict[str, Any] | None,
    min_frames: int,
) -> dict[str, Any]:
    """Prediction service state: how much nominal coverage exists and whether
    the model is active. The explanation, when requested, was computed by the
    calibrated service — never here."""
    coverage_pct = min(100.0, frame_count / _NOMINAL_FULL_COVERAGE_FRAMES * 100.0)
    model_ok = bool(service_status and service_status.get("active"))
    status = (
        explanation.get("status")
        if explanation
        else ("active" if (model_ok and frame_count >= min_frames) else ("collecting" if frame_count < min_frames else "physics_only_fallback"))
    )
    block: dict[str, Any] = {
        "provenance": "prediction",
        "available": model_ok,
        "status": status,
        "frame_count": frame_count,
        "min_frames": min_frames,
        "full_coverage_frames": _NOMINAL_FULL_COVERAGE_FRAMES,
        "coverage_pct": round(coverage_pct, 1),
        "service": service_status or {"status": "unreachable"},
        "explanation": explanation,
    }
    return block


def models_block(providers: dict[str, Any] | None, routes: dict[str, Any] | None, healthy_chain: list[str] | None) -> dict[str, Any]:
    """Gateway provider table — which models exist and which chain would serve."""
    return {
        "provenance": "derived",
        "providers": providers or {"error": "gateway unreachable"},
        "priority_chain": (routes or {}).get("priority_chain", []),
        "health": (routes or {}).get("health", {}),
        "healthy_chain": healthy_chain or [],
        "serving_note": "on this stack the assistant is answered by the mock provider.",
    }


def documents_block(index_info: dict[str, Any] | None) -> dict[str, Any]:
    """RAG index metadata. Text provenance is documented in each citation."""
    return {
        "provenance": "documentation",
        "index": index_info or {"error": "rag unreachable"},
        "label": "internal assistant guide — not the manufacturer's documentation",
    }


def services_block(gateway: str, rag: str, sim: str, assistant: str = "ok") -> dict[str, Any]:
    """Per-service readiness, labelled derived (probed, not computed)."""
    return {
        "provenance": "derived",
        "assistant": assistant,
        "gateway": gateway,
        "rag": rag,
        "simulator": sim,
    }


def identity_block(environment: str, simulated: bool) -> dict[str, Any]:
    return {
        "service": "assistant-api",
        "version": __version__,
        "prompt_version": PROMPT_VERSION,
        "rag_index_version": RAG_INDEX_VERSION,
        "environment": environment,
        "run": "simulated eval stack" if simulated else "instrumented vehicle",
        "telemetry_provenance": "simulator" if simulated else "measured",
    }


def assemble(
    *,
    now: float,
    identity: dict[str, Any],
    telemetry: dict[str, Any],
    proactive: dict[str, Any] | None,
    prediction: dict[str, Any],
    models: dict[str, Any],
    documents: dict[str, Any],
    services: dict[str, Any],
    scenario: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure composition of the HUD contract (unit-testable)."""
    block: dict[str, Any] = {
        "v": "v1",
        "generated_at": now,
        "vehicle": {
            "id": telemetry.get("vehicle_id"),
            "mode": telemetry.get("mode"),
            "charging": telemetry.get("charging"),
        },
        "telemetry": telemetry,
        "proactive": proactive,
        "prediction": prediction,
        "models": models,
        "documents": documents,
        "services": services,
        "identity": identity,
        "provenance_legend": PROVENANCE_LEGEND,
    }
    if scenario is not None:
        block["scenario"] = scenario
    if policy is not None:
        block["policy"] = policy
    if diagnostics is not None:
        block["diagnostics"] = diagnostics
    return block


async def collect_hud(pipeline: Any, evaluator: Any, include_explanation: bool = False) -> dict[str, Any]:
    """Gather frozen state from each owner and compose the HUD payload.

    External calls are best-effort: an unreachable owner yields an explicit
    ``error`` label in its block, never a fabricated number.
    """
    from app.config import settings  # lazy: keeps this module file-loadable under tests

    simulated = settings.environment in {"local", "ci", "demo"}
    telemetry = telemetry_block(pipeline.telemetry, simulated)

    gateway_url = settings.gateway_url
    rag_url = settings.rag_url
    sim_http = settings.telemetry_http

    async def _get(url: str) -> dict[str, Any]:
        try:
            resp = await pipeline._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - best-effort facade
            return {"error": f"unreachable ({exc.__class__.__name__})"}

    async def _timed_get(url: str) -> tuple[float | None, dict[str, Any]]:
        started = time.perf_counter()
        data = await _get(url)
        return (round((time.perf_counter() - started) * 1000.0, 2) if "error" not in data else None), data

    gateway_providers, gateway_routes, rag_info, pred_status = await asyncio.gather(
        _get(f"{gateway_url}/providers"),
        _get(f"{gateway_url}/routes"),
        _get(f"{rag_url}/index-info"),
        _get(f"{sim_http}/api/prediction/status"),
    )
    trip = await _get(f"{sim_http}/api/trip")
    control = await _get(f"{sim_http}/api/control")

    explanation = None
    if include_explanation:
        explanation = await pipeline._fetch_prediction_explanation()
        explanation = explanation.model_dump(mode="json") if explanation else None

    prediction = prediction_block(
        frame_count=len(pipeline.telemetry.frames),
        service_status=pred_status,
        explanation=explanation,
        min_frames=settings.prediction_min_frames,
    )
    if "error" not in trip:
        prediction["trip"] = trip

    healthy = [
        p
        for p, h in (gateway_routes.get("health", {}) if "error" not in gateway_routes else {}).items()
        if h.get("circuit") != "open"
    ]

    probes = await asyncio.gather(
        _get(f"{gateway_url}/healthz"),
        _get(f"{rag_url}/healthz"),
        _get(f"{sim_http}/healthz"),
    )

    def _status(resp: dict[str, Any]) -> str:
        return "ok" if "error" not in resp and resp.get("status") == "ok" else "degraded"

    provider_lat, provider_latency_probe = await _timed_get(f"{gateway_url}/providers")
    rag_lat, rag_latency_probe = await _timed_get(f"{rag_url}/index-info")
    sim_lat, sim_latency_probe = await _timed_get(f"{sim_http}/api/prediction/status")

    scenario = {"provenance": "simulator"}
    if "error" not in control and control:
        scenario["control"] = control
    else:
        scenario["error"] = "control unreachable"

    policy = {
        "provenance": "derived",
        "level": getattr(settings, "policy_level", "read_only"),
        "predictive_actions_enabled": bool(getattr(settings, "predictive_actions_enabled", False)),
        "proactive_poll_interval_s": getattr(evaluator, "poll_interval_s", None),
        "prediction_min_frames": settings.prediction_min_frames,
    }

    diagnostics = {
        "provenance": "derived",
        "latency_ms": {
            f"{gateway_url}/providers": provider_lat if "error" not in provider_latency_probe else None,
            f"{rag_url}/index-info": rag_lat if "error" not in rag_latency_probe else None,
            f"{sim_http}/api/prediction/status": sim_lat if "error" not in sim_latency_probe else None,
        },
        "probe": {
            "gateway_providers": provider_latency_probe.get("error") if "error" in provider_latency_probe else "ok",
            "rag_index": rag_latency_probe.get("error") if "error" in rag_latency_probe else "ok",
            "sim_control": control.get("error") if "error" in control else "ok",
        },
    }

    return assemble(
        now=time.time(),
        identity=identity_block(settings.environment, simulated),
        telemetry=telemetry,
        proactive=proactive_block(evaluator),
        prediction=prediction,
        models=models_block(gateway_providers if "error" not in gateway_providers else None, gateway_routes if "error" not in gateway_routes else None, healthy),
        documents=documents_block(rag_info if "error" not in rag_info else None),
        services=services_block(_status(probes[0]), _status(probes[1]), _status(probes[2])),
        scenario=scenario,
        policy=policy,
        diagnostics=diagnostics,
    )