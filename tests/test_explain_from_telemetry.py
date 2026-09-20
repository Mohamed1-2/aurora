"""WS3 frames -> explanation bridge tests (raw telemetry -> deterministic block).

This is the runtime path the assistant uses for "why" questions: it forwards
the 10 Hz frames it already grounds on, and the simulator bins them, re-derives
the canonical feature vector with the frozen extractor + WS1 physics, then
builds the deterministic explanation. Contracts here mirror
``test_calibration_explain`` but at the *raw frame* level, so the assistant
never has to re-derive features itself.

The simulator's ``app`` package is loaded lazily (never at module import) so it
never disturbs collection of other test modules that own their own ``app``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest

ARTIFACT_DIR = ROOT / "artifacts" / "calibration" / "v2-ws1"
HAS_ARTIFACT = ARTIFACT_DIR.exists() and (ARTIFACT_DIR / "calibration_model.lgb").exists()


def _load_explain_module():
    for _p in (str(ROOT), str(ROOT / "services" / "telemetry-simulator"), str(ROOT / "services")):
        sys.path.insert(0, _p)  # hoist unconditionally: assistant/gateway modules may have stashed their dir at [0]
    for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_mod, None)
    import importlib

    return importlib.import_module("app.engine.explain_from_telemetry")


def _frames(speed_kph: float, seconds: float = 320.0, hz: float = 10.0) -> list[dict]:
    n = int(seconds * hz)
    return [
        {
            "vehicle_id": "ORION-X-TEST-01",
            "episode_id": "test",
            "seq": i,
            "signals": {
                "vehicle_speed_kph": speed_kph,
                "hvac_power_kw": 1.5,
                # per-0.1s-tick Wh: power_W * dt / 3600 (sim convention)
                "hvac_energy_wh_tick": 1.5 / 36.0,
                "aux_energy_wh_tick": 0.35 / 36.0,
                "outside_temp_c": 12.0,
                "coolant_temp_c": 30.0,
                "battery_voltage_v": 350.0,
                "battery_soc_pct": 80.0 - (0.8 * i / n),
            },
        }
        for i in range(n)
    ]


def test_frames_bin_into_10s_with_live_signal_mapping():
    mod = _load_explain_module()
    bins = mod.frames_to_bins(_frames(80.0, seconds=100.0))
    assert len(bins) == 10
    b = bins[0]
    assert b["speed_kph"] == pytest.approx(80.0)
    # hvac 1.5 kW + aux 0.35 kW (from Wh per 0.1s tick) == 1850 W
    assert b["hv_aux_power_w"] == pytest.approx(1850.0, rel=1e-6)
    assert b["outside_temp_c"] == pytest.approx(12.0)
    assert b["coolant_temp_c"] == pytest.approx(30.0)
    assert b["pack_voltage_v"] == pytest.approx(350.0)


def test_features_from_frames_drops_absent_thermal_signals_to_none():
    mod = _load_explain_module()
    req = mod.features_from_frames(_frames(80.0))
    assert req is not None
    assert req.speed_kph_mean == pytest.approx(80.0)
    assert req.physics_soc_drop_pct > 0.0
    # the simulator does not emit pack in/outlet or stator/rotor -> honest None
    for field in ("battery_temp_inlet_c_mean", "battery_temp_outlet_c_mean",
                  "motor_stator_temp_c_mean", "motor_rotor_temp_c_mean"):
        assert getattr(req, field) is None


def test_features_from_frames_too_short_returns_none():
    mod = _load_explain_module()
    assert mod.features_from_frames(_frames(80.0, seconds=(mod._MIN_BINS - 1) * 10.0)) is None
    assert mod.features_from_frames([]) is None


skip_reason = pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")


@skip_reason
def test_explain_from_frames_absolute_identities_hold():
    mod = _load_explain_module()
    e = mod.explain_from_frames(_frames(80.0))
    assert e.status == "active"
    assert e.shape == "absolute"
    assert e.to_prediction.calibrated_soc_drop_pct > 0.0
    chk = e.identity_checks
    assert chk.physics_sum_matches is True
    assert chk.ml_sum_matches is True
    assert chk.blend_matches is True
    assert e.top_causes  # ranked, code-computed
    assert all(c.provenance in ("physics", "ml") for c in e.top_causes)


@skip_reason
def test_explain_from_frames_delta_uses_both_windows():
    mod = _load_explain_module()
    e = mod.explain_from_frames(_frames(80.0), from_frames=_frames(40.0))
    assert e.shape == "delta"
    assert e.from_prediction is not None
    assert e.to_prediction.calibrated_soc_drop_pct > e.from_prediction.calibrated_soc_drop_pct
    assert e.identity_checks.delta_matches is True


def test_explain_from_frames_never_raises_and_falls_back_cleanly():
    mod = _load_explain_module()
    # too few frames -> physics-only fallback, not an exception
    e = mod.explain_from_frames(_frames(80.0, seconds=10.0))
    assert e.status == "physics_only_fallback"
    assert e.fallback_reason
    assert e.to_prediction.calibrated_soc_drop_pct == 0.0  # no physics computed on 1 bin
    # empty input -> same fallback contract
    e2 = mod.explain_from_frames([])
    assert e2.status == "physics_only_fallback"


def test_replay_snapshot_single_frame_is_graceful():
    """Replay fallback pushes ONE frame without signals -- never hard-fails."""
    mod = _load_explain_module()
    e = mod.explain_from_frames([{"signals": {"battery_soc_pct": 80.0}}])
    assert e.status == "physics_only_fallback"