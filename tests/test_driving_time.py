"""Deterministic "how long can I drive" estimator tests.

Cover the six mandated behaviours for the no-LLM driving-time path:

  1. normal moving vehicle   -> range / representative speed
  2. stopped vehicle          -> calibrated discharge forecast (prediction-backed)
  3. missing range/data       -> honest refusal, no invented number
  4. prediction-backed estimate -> SOC pts-per-5-min math, works at standstill
  5. no divide-by-zero        -> soc=0 / drop=0 / speed=0 guards
  6. grounded answer          -> every numeric token in the answer is one of the
                                code-produced values (never an LLM invention)

The assistant's ``app`` package is lazy-loaded exactly like
``test_assistant_prediction_fetch`` so other suites own their own ``app``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest

from vehicle_ai.schemas import PredictionExplanation  # noqa: E402


def _load_assistant(module: str):
    for _p in (str(ROOT), str(ROOT / "services" / "assistant-api")):
        sys.path.insert(0, _p)
    for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_mod, None)
    import importlib

    return importlib.import_module(module)


def _dt_mod():
    return _load_assistant("app.core.driving_time")


# ---------------------------------------------------------------------------
# Grounding / prediction fixtures
# ---------------------------------------------------------------------------


def _grounding(signals: dict | None = None, mean: dict | None = None, charging: bool = False) -> dict:
    return {
        "vehicle_id": "ORION-X-TEST-01",
        "timestamp": "2026-01-01T00:00:00Z",
        "mode": "normal",
        "charging": charging,
        "signals": signals or {},
        "mean": mean or {},
        "min": {},
        "max": {},
        "trending": {},
        "dtcs": [],
        "warnings": [],
        "source": "simulator",
    }


def _explanation(drop_pct: float | None = 3.9) -> PredictionExplanation:
    return PredictionExplanation.model_validate(
        {
            "shape": "absolute",
            "status": "active",
            "model_version": "v2-ws1-r1",
            "capacity_kwh": 58.0,
            "to_prediction": {
                "physics_soc_drop_pct": drop_pct,
                "ml_residual_pct": 0.0,
                "calibrated_soc_drop_pct": drop_pct,
                "speed_band": "highway",
                "lambda_applied": 1.0,
                "model_version": "v2-ws1-r1",
                "status": "active",
            },
            "from_prediction": None,
            "physics_contributions": [],
            "ml_contributions": [],
            "top_causes": [],
            "identity_checks": {},
        }
    )


# ---------------------------------------------------------------------------
# 1 + 2 + 4 — estimate correctness
# ---------------------------------------------------------------------------


def test_normal_moving_vehicle_uses_range_over_speed():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 80.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 64.0},
            mean={"vehicle_speed_kph": 80.0},
        )
    )
    assert est.estimable and not est.prediction_backed
    assert est.speed_kph == 80.0
    assert est.range_km == 100.0
    assert est.minutes == pytest.approx(75.0)  # 100 km / 80 km/h * 60
    assert est.refusals == []


def test_stopped_vehicle_is_prediction_backed():
    """Standstill, but the calibrated discharge forecast still yields a time."""
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 120.0, "battery_soc_pct": 64.0},
            mean={"vehicle_speed_kph": 0.0},
        ),
        explanation=_explanation(drop_pct=3.9),
    )
    assert est.estimable and est.prediction_backed
    assert est.minutes == pytest.approx((64.0 / 3.9) * 5.0)
    assert est.drop_pct_per_5min == pytest.approx(3.9)


def test_prediction_backed_priority_over_range():
    """When a forecast exists it wins, even if range/speed also available."""
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 80.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 50.0},
            mean={"vehicle_speed_kph": 80.0},
        ),
        explanation=_explanation(drop_pct=2.0),
    )
    assert est.prediction_backed
    assert est.minutes == pytest.approx((50.0 / 2.0) * 5.0)  # 125 min, not 75


def test_arrival_eta_included_when_present():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={**{"vehicle_speed_kph": 80.0, "distance_to_empty_km": 100.0}, "destination_remaining_km": 20.0, "arrival_eta_min": 15.0},
            mean={"vehicle_speed_kph": 80.0},
        )
    )
    assert est.arrival_eta_min == 15.0
    out = _dt_mod().render_driving_time(est)
    assert "destination is about 15 min away" in out


# ---------------------------------------------------------------------------
# 3 — missing data => honest refusal
# ---------------------------------------------------------------------------


def test_empty_grounding_refuses_honestly():
    est = _dt_mod().estimate_driving_time({})
    assert not est.estimable
    assert est.refusals and "no usable range signal" in est.refusals[0]
    out = _dt_mod().render_driving_refusal(est)
    assert "can't give a reliable driving-time estimate" in out
    assert re.search(r"\d", out) is None  # no invented numbers in the refusal


def test_missing_frames_path_grounding_refuses():
    # what grounding() actually returns for an empty frame buffer
    est = _dt_mod().estimate_driving_time(
        {"vehicle_id": "ORION-X-TEST-01", "signals": {}, "mean": {}, "charging": False, "mode": ""}
    )
    assert not est.estimable
    assert est.refusals


def test_stopped_without_forecast_refuses():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 120.0},
            mean={"vehicle_speed_kph": 0.0},
        )
    )
    assert not est.estimable
    assert "discharge forecast" in est.refusals[0]


def test_zero_range_refuses_honestly():
    # live sim state: SOC 0 / range 0 — an estimate of "1 min" would be silly
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 80.0, "distance_to_empty_km": 0.0, "battery_soc_pct": 0.0},
            mean={"vehicle_speed_kph": 80.0},
        ),
        explanation=_explanation(drop_pct=3.9),
    )
    assert not est.estimable
    assert "zero remaining range" in est.refusals[0]
    out = _dt_mod().render_driving_refusal(est)
    assert "zero remaining range" in out


def test_charging_refuses():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 64.0},
            mean={"vehicle_speed_kph": 0.0},
            charging=True,
        ),
        explanation=_explanation(drop_pct=3.9),
    )
    assert not est.estimable and not est.prediction_backed
    assert "charging" in est.refusals[0]


# ---------------------------------------------------------------------------
# 5 — no divide-by-zero anywhere
# ---------------------------------------------------------------------------


def test_zero_speed_and_zero_forecast_never_divide_by_zero():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 60.0},
            mean={"vehicle_speed_kph": 0.0},
        ),
        explanation=_explanation(drop_pct=0.0),  # degenerate forecast
    )
    # 0.0 drop is not "positive", and speed 0 can't form a range estimate => honest refusal
    assert est.minutes is None and not est.estimable
    assert est.drop_pct_per_5min is None
    assert est.refusals


def test_zero_soc_with_forecast_never_divide_by_zero():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 0.0},
            mean={"vehicle_speed_kph": 0.0},
        ),
        explanation=_explanation(drop_pct=3.9),
    )
    assert est.minutes is None  # never 0.0/3.9*5 -> 0 or NaN
    assert est.soc_pct == 0.0
    assert est.refusals


def test_zero_drop_falls_back_to_range_path():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 90.0, "distance_to_empty_km": 150.0, "battery_soc_pct": 60.0},
            mean={"vehicle_speed_kph": 90.0},
        ),
        explanation=_explanation(drop_pct=0.0),
    )
    assert est.estimable and not est.prediction_backed
    assert est.minutes == pytest.approx((150.0 / 90.0) * 60.0)


# ---------------------------------------------------------------------------
# 6 — answer numbers all code-produced
# ---------------------------------------------------------------------------


def _rendered_numbers(est) -> set[int]:
    """Mirror the renderer: every number in an answer must be a digit-substring
    of a rendered estimate field or the fixed prediction-block constant.

    This is the "only code-produced numbers" guarantee: a hallucinated integer
    (e.g. 95 min when SOC says 64) is not a substring of any measured value.
    """
    numbers: set[int] = set()

    def add(s: str) -> None:
        if not re.search(r"\d", s):
            return
        for m in re.finditer(r"-?\d+(?:\.\d+)?", s):
            text = m.group()
            sign = -1 if text.startswith("-") else 1
            clean = text.lstrip("-")
            for i in range(len(clean)):
                for j in range(i + 1, len(clean) + 1):
                    sub = clean[i:j]
                    if sub.isdigit():  # skip slices that cross the decimal point
                        numbers.add(sign * int(sub))
            numbers.add(sign * int(float(text)))

    if est.minutes is not None:
        m = max(1, int(round(est.minutes)))
        h, mm = divmod(m, 60)
        add(f"{h} {mm:02d}")  # match _minutes_human components
    for v in (est.range_km, est.speed_kph, est.soc_pct, est.arrival_eta_min):
        if v is not None:
            add(f"{v:g}")
    if est.drop_pct_per_5min is not None:
        add(f"{est.drop_pct_per_5min:.2f}")  # renderer formats with :.2f
    # prediction-block clock the forecast is calibrated against
    add(f"{_dt_mod()._BLOCK_MINUTES:g}")
    return numbers


def test_grounded_answer_contains_only_code_produced_numbers():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 80.0, "distance_to_empty_km": 100.0, "battery_soc_pct": 64.0},
            mean={"vehicle_speed_kph": 80.0},
        )
    )
    out = _dt_mod().render_driving_time(est)
    tokens = [int(t) for t in re.findall(r"\d+", out)]
    assert tokens, "answer should carry at least one number"
    allowed = _rendered_numbers(est)
    assert all(t in allowed for t in tokens), f"unproduced numbers {set(tokens) - allowed} in: {out}"


def test_prediction_backed_answer_numbers_code_produced():
    est = _dt_mod().estimate_driving_time(
        _grounding(
            signals={"vehicle_speed_kph": 0.0, "distance_to_empty_km": 120.0, "battery_soc_pct": 64.0},
            mean={"vehicle_speed_kph": 0.0},
        ),
        explanation=_explanation(drop_pct=3.9),
    )
    out = _dt_mod().render_driving_time(est)
    tokens = [int(t) for t in re.findall(r"\d+", out)]
    allowed = _rendered_numbers(est)
    assert all(t in allowed for t in tokens)
    # spot-check: the estimate really is 64% / (3.9 pts per 5 min)
    assert "1 h 22 min" in out