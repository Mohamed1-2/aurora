"""WS4 proactive engine unit tests: strict signal-gating, thresholds,
cooldown/suppression, monotonic severity, resolution and recurrence.

The proactive engine is pure code (no LLM). These tests exercise the state
machine with a fake telemetry source. The assistant ``app`` package is loaded
lazily inside the loader (never at module import) so collection of OTHER test
modules that own their own ``app`` (gateway, simulator) is never disturbed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest
from vehicle_ai.schemas import CalibratedSoCDropResponse, PredictionExplanation, Severity  # noqa: E402


def _load(module: str):
    for _p in (str(ROOT), str(ROOT / "services" / "assistant-api")):
        sys.path.insert(0, _p)
    for _k in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_k, None)
    import importlib

    return importlib.import_module(module)


def _engine_module():
    return _load("app.core.proactive_engine")


class FakeTelemetry:
    """Scriptable stand-in for TelemetryIngest (grounding/series/frames/has_signal)."""

    def __init__(self, signals=None, series=None, charging=False, frames=None):
        self._signals = signals or {}
        self._series = series or {}
        self._charging = charging
        if frames is not None:
            self.frames = list(frames)
        else:
            self.frames = [{"vehicle_id": "v", "signals": dict(self._signals)}]

    def has_signal(self, sig: str) -> bool:
        return sig in self._signals

    def signal_series(self, sig: str, max_age_s: float = 45.0, stride_s: float = 1.0):
        return self._series.get(sig, [])

    def grounding(self):
        latest = self.frames[-1] if self.frames else {}
        return {
            "vehicle_id": latest.get("vehicle_id", "v"),
            "signals": dict(self._signals),
            "charging": self._charging,
            "dtcs": [d for d in latest.get("active_dtcs", [])],
        }


def _mk_frames(n: int, active_dtcs=None) -> list[dict]:
    f = {"vehicle_id": "v", "seq": 0, "active_dtcs": active_dtcs or []}
    return [dict(f, seq=i) for i in range(n)]


PAST = time.time() - 30.0
SEC = 1.0


def _series(start: float, step: float, n: int = 30) -> list[tuple[float, float]]:
    return [(PAST + i * SEC, start + i * step) for i in range(n)]


def _discharge_explanation(status: str = "active", drop: float = 6.0) -> PredictionExplanation:
    return PredictionExplanation(
        shape="absolute",
        status=status,
        fallback_reason="" if status == "active" else "ml-artifact-unavailable",
        to_prediction=CalibratedSoCDropResponse(
            physics_soc_drop_pct=drop - 1.0,
            ml_residual_pct=1.0,
            calibrated_soc_drop_pct=drop,
            speed_band="medium",
            lambda_applied=1.0,
            model_version="v2-ws1-r1" if status == "active" else None,
            status=status,
        ),
        capacity_kwh=58.0,
    )


@pytest.fixture()
def eng():
    return _engine_module()


# ---------------------------------------------------------------------------
# Strict signal-gating: missing signal -> no inference, detector skipped
# ---------------------------------------------------------------------------


def test_missing_tpms_means_no_leak_inference(eng):
    evaluator = eng.ProactiveEvaluator(FakeTelemetry(signals={"battery_soc_pct": 50.0}), get_explanation=None)
    snap = _poll(evaluator)
    assert not snap.active
    assert "tire_pressure_leak" in snap.detectors_skipped_no_signal


def test_missing_range_means_no_low_range_alert(eng):
    evaluator = eng.ProactiveEvaluator(FakeTelemetry(signals={"battery_temp_c": 20.0}))
    snap = _poll(evaluator)
    assert not snap.active
    assert "low_range" in snap.detectors_skipped_no_signal


def test_missing_dtc_field_means_no_dtc_inference(eng):
    frames = _mk_frames(5)  # note: no "active_dtcs" key at all
    tele = FakeTelemetry(signals={"battery_soc_pct": 50.0}, frames=frames)
    tele.frames = [{"source": "replay", "signals": {}}]
    evaluator = eng.ProactiveEvaluator(tele)
    snap = _poll(evaluator)
    assert snap.active == []


# ---------------------------------------------------------------------------
# Detector thresholds
# ---------------------------------------------------------------------------


def test_low_range_fires_below_threshold_only(eng):
    t = FakeTelemetry(signals={"distance_to_empty_km": 35.0, "battery_soc_pct": 40.0})
    ev = eng.ProactiveEvaluator(t)
    _poll(ev); _poll(ev)  # confirm_polls=2
    snap = _poll(ev)
    types = [e.type for e in snap.active]
    assert "low_range" in types


def test_low_range_suppressed_while_charging(eng):
    t = FakeTelemetry(signals={"distance_to_empty_km": 20.0}, charging=True)
    ev = eng.ProactiveEvaluator(t)
    for _ in range(5):
        _poll(ev)
    assert all(e.type != "low_range" for e in ev.snapshot().active)


def test_low_range_not_fired_above_threshold(eng):
    t = FakeTelemetry(signals={"distance_to_empty_km": 120.0})
    ev = eng.ProactiveEvaluator(t)
    for _ in range(5):
        _poll(ev)
    assert ev.snapshot().active == []


def test_tire_leak_fires_on_sustained_slope(eng):
    series = {"tire_pressure_rr_kpa": _series(240.0, -0.8)}  # -48 kPa/min over 30 s
    signals = {"tire_pressure_fl_kpa": 240.0, "tire_pressure_fr_kpa": 240.0,
               "tire_pressure_rl_kpa": 240.0, "tire_pressure_rr_kpa": 240.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series))
    _poll(ev); _poll(ev)
    snap = _poll(ev)
    leak = [e for e in snap.recent if e.type == "tire_pressure_leak"]
    assert leak


def test_tire_leak_trigger_is_a_slope_not_a_pressure(eng):
    # A change-rate alert must be evidence about the *rate* (kPa/min), with the
    # raw pressure carried separately — the trigger signal must never be the
    # raw wheel pressure while observed is a rate.
    series = {"tire_pressure_rr_kpa": _series(240.0, -0.8)}
    signals = {"tire_pressure_fl_kpa": 240.0, "tire_pressure_fr_kpa": 240.0,
               "tire_pressure_rl_kpa": 240.0, "tire_pressure_rr_kpa": 240.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series))
    _poll(ev); _poll(ev)
    snap = _poll(ev)
    leak = [e for e in snap.recent if e.type == "tire_pressure_leak"][0]

    assert leak.triggers and leak.triggers[0].signal == "tire_pressure_rr_kpa_slope_per_min"
    assert leak.triggers[0].unit == "kPa/min"
    assert leak.triggers[0].threshold == -12.0
    assert leak.triggers[0].observed < -12.0

    # Raw pressure and its change rate are SEPARATE evidence fields.
    by_signal = {e.signal: e for e in leak.evidence}
    assert "tire_pressure_rr_kpa" in by_signal           # pressure, in kPa
    assert by_signal["tire_pressure_rr_kpa"].unit == "kPa"
    assert "tire_pressure_rr_kpa_slope_per_min" in by_signal  # rate, in kPa/min
    assert by_signal["tire_pressure_rr_kpa_slope_per_min"].unit == "kPa/min"
    assert by_signal["tire_pressure_rr_kpa_slope_per_min"].value < -12.0


def test_tire_no_alert_on_steady_pressure(eng):
    series = {"tire_pressure_rr_kpa": _series(240.0, 0.0)}
    signals = {"tire_pressure_fl_kpa": 240.0, "tire_pressure_fr_kpa": 240.0,
               "tire_pressure_rl_kpa": 240.0, "tire_pressure_rr_kpa": 240.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series))
    for _ in range(5):
        _poll(ev)
    snap = ev.snapshot()
    assert all(e.type != "tire_pressure_leak" for e in snap.active + snap.recent)


def test_battery_overtemperature_and_escalation(eng):
    signals = {"battery_temp_c": 48.0}
    series = {"battery_temp_c": _series(48.0, 0.0)}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series))
    _poll(ev); _poll(ev); _poll(ev)
    active = ev.snapshot().active
    assert any(e.type == "battery_overtemperature" and e.severity == Severity.WARNING for e in active)

    series = {"battery_temp_c": _series(62.0, 0.0)}
    signals = {"battery_temp_c": 62.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series))
    snap = _poll(ev); snap = _poll(ev); snap = _poll(ev)
    crit = [e for e in snap.active if e.type == "battery_overtemperature"]
    assert crit and crit[0].severity == Severity.CRITICAL
    assert "thermal" in crit[0].title.lower()


def test_charging_stall_fires_only_when_plugged_in_with_zero_rate(eng):
    signals = {"charge_rate_kw": 0.0, "battery_soc_pct": 30.0}
    series = {"charge_rate_kw": _series(0.0, 0.0)}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series, charging=True))
    _poll(ev); _poll(ev); _poll(ev)
    assert any(e.type == "charging_stall" for e in ev.snapshot().active)

    ev2 = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series, charging=False))
    for _ in range(5):
        _poll(ev2)
    assert all(e.type != "charging_stall" for e in ev2.snapshot().active)


def test_charging_healthy_flow_never_alerts(eng):
    signals = {"charge_rate_kw": 11.0}
    series = {"charge_rate_kw": _series(11.0, 0.0)}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, series=series, charging=True))
    for _ in range(5):
        _poll(ev)
    assert all(e.type != "charging_stall" for e in ev.snapshot().active)


def test_dtc_new_critical_code_fires_critical_event(eng):
    frames = _mk_frames(5, active_dtcs=[
        {"code": "C0031", "description": "ABS degraded", "severity": "critical"}
    ])
    signals = {"battery_soc_pct": 50.0}
    tele = FakeTelemetry(signals=signals, frames=frames)
    ev = eng.ProactiveEvaluator(tele)
    snap = _poll(ev)
    assert any(e.type == "dtc_escalation" and e.severity == Severity.CRITICAL for e in snap.recent)


def test_dtc_stable_code_does_not_refire(eng):
    frames = _mk_frames(5, active_dtcs=[
        {"code": "P0420", "description": "catalyst below threshold", "severity": "warning"}
    ])
    tele = FakeTelemetry(signals={"battery_soc_pct": 50.0}, frames=frames)
    ev = eng.ProactiveEvaluator(tele)
    for _ in range(4):
        _poll(ev)
    snaps = ev.snapshot()
    dtc = [e for e in snaps.active + snaps.recent if e.type == "dtc_escalation"]
    assert dtc and dtc[0].occurrence_count == 1


# ---------------------------------------------------------------------------
# Prediction / data availability guardrails
# ---------------------------------------------------------------------------


def test_discharge_risk_skipped_without_nominal_coverage(eng):
    signals = {s: 40.0 for s in ["vehicle_speed_kph", "battery_soc_pct", "outside_temp_c",
                                 "coolant_temp_c", "battery_voltage_v", "hvac_power_kw"]}
    short = _mk_frames(100)
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals, frames=short), get_explanation=lambda: _discharge_explanation())
    snap = _poll(ev)
    assert "prediction_discharge_risk" in snap.detectors_skipped_no_signal


def test_discharge_risk_full_status_when_active_model(eng):
    signals = {s: 40.0 for s in ["vehicle_speed_kph", "battery_soc_pct", "outside_temp_c",
                                 "coolant_temp_c", "battery_voltage_v", "hvac_power_kw"]}
    long = _mk_frames(3010)
    ev = eng.ProactiveEvaluator(
        FakeTelemetry(signals=signals, frames=long),
        get_explanation=lambda: _discharge_explanation(status="active", drop=6.0),
    )
    _poll(ev); _poll(ev)
    snap = _poll(ev)
    evs = [e for e in snap.active if e.type == "prediction_discharge_risk"]
    assert evs and evs[0].status == "full"


def test_discharge_risk_fallback_downgrades_and_marks_physics(eng):
    signals = {s: 40.0 for s in ["vehicle_speed_kph", "battery_soc_pct", "outside_temp_c",
                                 "coolant_temp_c", "battery_voltage_v", "hvac_power_kw"]}
    long = _mk_frames(3010)
    ev = eng.ProactiveEvaluator(
        FakeTelemetry(signals=signals, frames=long),
        get_explanation=lambda: _discharge_explanation(status="physics_only_fallback", drop=6.0),
    )
    _poll(ev); _poll(ev)
    snap = _poll(ev)
    evs = [e for e in snap.active if e.type == "prediction_discharge_risk"]
    assert evs and evs[0].status == "fallback" and "physics estimate" in evs[0].summary.lower()


# ---------------------------------------------------------------------------
# Cooldown, suppression, resolution, recurrence
# ---------------------------------------------------------------------------


def test_confirmation_requires_consecutive_polls(eng):
    signals = {"distance_to_empty_km": 30.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals))
    _poll(ev)  # 1/2 confirmations
    assert not ev.snapshot().active
    _poll(ev)
    assert any(e.type == "low_range" for e in ev.snapshot().active)


def test_cooldown_suppresses_duplicate_while_active(eng):
    signals = {"distance_to_empty_km": 30.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals))
    for _ in range(5):
        _poll(ev)
    snap = ev.snapshot()
    lr = [e for e in snap.active if e.type == "low_range"]
    assert lr[0].occurrence_count == 1
    assert ev.stats["suppressed"] >= 1
    assert ev.stats["emitted"] == 1


def test_resolution_removes_after_clear_and_recurrence_reattempts(eng):
    signals = {"distance_to_empty_km": 30.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals))
    for _ in range(3):
        _poll(ev)
    assert any(e.type == "low_range" for e in ev.snapshot().active)

    # Range restored -> detector no longer fires -> resolution after 3 polls.
    ev.telemetry._signals = {"distance_to_empty_km": 200.0}
    for _ in range(4):
        _poll(ev)
    assert all(e.type != "low_range" for e in ev.snapshot().active)
    resolved = [e for e in ev.snapshot().recent if e.type == "low_range"]
    assert resolved and resolved[0].occurrence_count == 1

# Recurrence (range drops again) -> fresh episode, new emission.
    ev.telemetry._signals = {"distance_to_empty_km": 25.0}
    for _ in range(4):
        _poll(ev)
    s = ev.snapshot()
    recurred = [e for e in s.active if e.type == "low_range"]
    assert recurred and recurred[0].occurrence_count == 2
    assert any(e.type == "low_range" and e.occurrence_count == 2 for e in s.recent)


def test_ack_dismisses_until_recovery(eng):
    signals = {"distance_to_empty_km": 30.0}
    ev = eng.ProactiveEvaluator(FakeTelemetry(signals=signals))
    for _ in range(3):
        _poll(ev)
    assert ev.top_active(1)
    ev.ack([ev.top_active(1)[0].event_id])
    assert ev.top_active(1) == []
    assert ev.snapshot().active == []


def _poll(ev):
    import asyncio

    return asyncio.run(ev.poll())