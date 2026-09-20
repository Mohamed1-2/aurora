"""Deterministic "how much driving time is left" estimator (no LLM).

Every number in the answer is produced by code from either the grounded
telemetry or the prediction engine's calibrated discharge forecast. The LLM is
never asked to calculate or invent any of these numbers — this module builds a
complete, honest answer on its own.

If the required signals are genuinely unavailable (no range, no usable speed,
no discharge forecast), the estimator refuses honestly instead of guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Prediction-block length the forecast is calibrated against (SOC pts per this
# many minutes). Kept as a named constant so every number in a rendered answer
# derives from either this or the measured/computed signals — test-enforced.
_BLOCK_MINUTES = 5.0

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class DrivingTimeEstimate:
    """Ready-to-render estimate, or the list of honest refusal reasons."""

    minutes: float | None = None
    range_km: float | None = None
    speed_kph: float | None = None
    soc_pct: float | None = None
    drop_pct_per_5min: float | None = None
    prediction_backed: bool = False
    arrival_eta_min: float | None = None
    charging: bool = False
    assumptions: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def estimable(self) -> bool:
        return self.minutes is not None and not self.refusals


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Parse a number-like signal value; anything unusable becomes None."""

    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def _prediction_drop(explanation: Any) -> float | None:
    """Extract the calibrated SOC-drop (pts / 5 min) from a prediction object.

    Accepts either the shared ``PredictionExplanation`` model or already-dumped
    dict. Any malformed/missing payload yields None so the estimator can fall
    back to range + speed.
    """
    if explanation is None:
        return None
    to = getattr(explanation, "to_prediction", None) if not isinstance(explanation, dict) else (explanation.get("to_prediction") or {})
    if to is None:
        return None
    drop = getattr(to, "calibrated_soc_drop_pct", None) if not isinstance(to, dict) else to.get("calibrated_soc_drop_pct")
    if isinstance(to, dict):
        return _num(drop)
    if not isinstance(drop, (int, float)):  # pydantic model attribute
        return None
    return _num(drop)


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def estimate_driving_time(grounding: dict[str, Any], explanation: Any = None) -> DrivingTimeEstimate:
    """Estimate remaining driving time from grounded signals + prediction.

    Priority:
      1. calibrated discharge forecast (SOC pts / 5 min) — works at standstill
      2. remaining range / representative speed — requires motion
      3. honest refusal — the signals really are absent

    ``grounding`` is the assistant's telemetry grounding dict
    (``ingest.TelemetryIngest.grounding()``). ``explanation`` is the optional
    deterministic ``PredictionExplanation`` (or dict) from the simulation.
    """
    signals = grounding.get("signals") or {}
    mean = grounding.get("mean") or {}

    est = DrivingTimeEstimate(
        charging=bool(grounding.get("charging", False)),
        range_km=_num(signals.get("distance_to_empty_km")),
        soc_pct=_num(signals.get("battery_soc_pct")),
    )
    speed_now = _num(signals.get("vehicle_speed_kph"))
    speed_mean = _num(mean.get("vehicle_speed_kph"))
    est.speed_kph = speed_mean if _positive(speed_mean) else (speed_now if _positive(speed_now) else None)

    remaining_km = _num(signals.get("destination_remaining_km"))
    eta_min = _num(signals.get("arrival_eta_min"))
    if remaining_km is not None and remaining_km >= 0 and eta_min is not None and eta_min >= 0:
        est.arrival_eta_min = eta_min

    if est.charging:
        est.refusals.append("the car is charging, so remaining driving time is growing as the battery fills")
        return est

    drop = _prediction_drop(explanation)
    if _positive(drop) and est.soc_pct is not None and est.soc_pct > 0:
        est.drop_pct_per_5min = drop
        est.minutes = (est.soc_pct / drop) * _BLOCK_MINUTES  # pts at a pts-per-block rate
        est.prediction_backed = True
        est.assumptions.append(f"the onboard forecast holds at {drop:.2f} SOC pts per {_BLOCK_MINUTES:g} minutes")
        return est

    if est.range_km is not None and est.range_km <= 0:
        est.refusals.append("the vehicle reports zero remaining range")
        return est

    if est.range_km is not None and est.range_km >= 0 and _positive(est.speed_kph):
        est.minutes = est.range_km / est.speed_kph * 60.0
        est.assumptions.append(
            f"the {est.range_km:g} km of range is consumed at a representative {est.speed_kph:g} km/h"
        )
        est.assumptions.append("steady conditions — elevation, climate and traffic will change it")
        return est

    if est.range_km is None:
        est.refusals.append("no usable range signal (distance_to_empty_km) is available")
    elif est.speed_kph is None:
        est.refusals.append(
            "the car is stopped and I have no discharge forecast to convert range into time"
        )
    return est


# ---------------------------------------------------------------------------
# Rendering (numbers come only from the estimate above)
# ---------------------------------------------------------------------------


def _minutes_human(minutes: float) -> str:
    m = max(1, int(round(minutes)))
    h, mm = divmod(m, 60)
    return f"{m} min" if h == 0 else f"{h} h {mm:02d} min"


def render_driving_time(est: DrivingTimeEstimate) -> str:
    if est.prediction_backed:
        body = (
            f"Estimate: about {_minutes_human(est.minutes)} of driving time left — "
            f"that's {est.soc_pct:g}% SOC at a forecast discharge of {est.drop_pct_per_5min:.2f} "
            f"SOC pts per {_BLOCK_MINUTES:g} minutes."
        )
    else:
        body = (
            f"Estimate: about {_minutes_human(est.minutes)} of driving time left at roughly "
            f"{est.speed_kph:g} km/h over the ~{est.range_km:g} km of range in the telemetry."
        )
    lines = [body]
    if est.assumptions:
        lines.append("Assumptions: " + " · ".join(est.assumptions) + ".")
    if est.arrival_eta_min is not None:
        lines.append(f"Separately, your current destination is about {_minutes_human(est.arrival_eta_min)} away.")
    return "\n".join(lines)


def render_driving_refusal(est: DrivingTimeEstimate | None = None) -> str:
    if est and est.refusals:
        reason = est.refusals[0]
    else:
        reason = "the required driving-time signals are unavailable"
    return f"I can't give a reliable driving-time estimate right now — {reason}."