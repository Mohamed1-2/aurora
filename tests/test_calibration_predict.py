"""WS2 calibration prediction interface tests (frozen v2-ws1-r1).

Covers the runtime contract that was promoted:
* ``CalibratedSoCDropPredictor`` loads the frozen model + lambda policy.
* Per-band lambda shrinkage is applied (0-30 band shrinks; 60-90 full).
* Missing/None features degrade to NaN, never crash.
* Unavailable artifact => physics-only fallback with explicit status.
* The simulator service wrapper and endpoint behave the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

import pytest

ARTIFACT_DIR = ROOT / "artifacts" / "calibration" / "v2-ws1"
HAS_ARTIFACT = ARTIFACT_DIR.exists() and (ARTIFACT_DIR / "calibration_model.lgb").exists()

from calibration.predict import CalibratedSoCDropPredictor, LAMBDA_POLICY_PCT  # noqa: E402


def _features(speed_kph_mean: float = 80.0, physics: float = 2.5, **overrides) -> dict:
    f = {
        "speed_kph_mean": speed_kph_mean,
        "speed_kph_std": 5.0,
        "speed_kph_max": speed_kph_mean + 10.0,
        "speed_kph_p90": speed_kph_mean + 5.0,
        "distance_km": speed_kph_mean * 300.0 / 3600.0,
        "accel_mean_mps2": 0.0,
        "decel_fraction": 0.1,
        "regen_decel_energy_kwh": 0.02,
        "hv_aux_power_w_mean": 1200.0,
        "outside_temp_c_mean": 12.0,
        "coolant_temp_c_mean": 30.0,
        "battery_temp_inlet_c_mean": 24.0,
        "battery_temp_outlet_c_mean": 26.0,
        "motor_stator_temp_c_mean": 45.0,
        "motor_rotor_temp_c_mean": 50.0,
        "pack_voltage_v_mean": 350.0,
        "physics_soc_drop_pct": physics,
    }
    f.update(overrides)
    return f


# --- predictor: active path (frozen artifact present) -----------------------


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_predictor_loads_frozen_model_and_metadata():
    p = CalibratedSoCDropPredictor(ARTIFACT_DIR)
    assert p.available
    assert p.version == "v2-ws1-r1"
    assert p.error is None
    meta = p.meta or {}
    assert meta["best_iteration"] == 40


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_predict_active_path_applies_band_lambda():
    p = CalibratedSoCDropPredictor(ARTIFACT_DIR)
    # 60-90 band: lambda=1.0 => calibrated = physics + full residual
    slow_mid = p.predict(_features(speed_kph_mean=40.0, physics=1.5))  # 30-60, lam 1.0
    assert slow_mid.status == "active"
    assert slow_mid.lambda_applied == pytest.approx(1.0)
    assert slow_mid.calibrated_soc_drop_pct == pytest.approx(
        slow_mid.physics_soc_drop_pct + slow_mid.ml_residual_pct, rel=1e-9)

    # 0-30 band: lambda=0.20 => correction is shrunk toward physics
    slow = p.predict(_features(speed_kph_mean=20.0, physics=1.0))
    assert slow.band == "0-30"
    assert slow.lambda_applied == pytest.approx(0.20)
    assert slow.calibrated_soc_drop_pct == pytest.approx(
        slow.physics_soc_drop_pct + 0.20 * slow.ml_residual_pct, rel=1e-9)


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_predict_none_features_are_tolerated_as_nan():
    p = CalibratedSoCDropPredictor(ARTIFACT_DIR)
    f = _features(speed_kph_mean=70.0)
    f["coolant_temp_c_mean"] = None
    f["motor_stator_temp_c_mean"] = None
    res = p.predict(f)
    assert res.status == "active"
    assert res.calibrated_soc_drop_pct == res.calibrated_soc_drop_pct  # not NaN
    assert res.band == "60-90"


# --- predictor: fallback path (artifact unavailable) -------------------------


def test_predictor_fallback_physics_only_when_artifact_missing():
    p = CalibratedSoCDropPredictor(ROOT / "artifacts" / "calibration" / "does-not-exist")
    assert not p.available
    res = p.predict(_features(speed_kph_mean=60.0, physics=2.0))
    assert res.status == "physics_only_fallback"
    assert res.calibrated_soc_drop_pct == pytest.approx(res.physics_soc_drop_pct)
    assert res.ml_residual_pct == 0.0
    assert res.fallback_reason


def test_lambda_policy_frozen():
    assert LAMBDA_POLICY_PCT == {"0-30": 0.20, "30-60": 1.00, "60-90": 1.00,
                                 "90-120": 1.00, "120+": 0.55}


# --- service wrapper + endpoint ----------------------------------------------


def _load_sim_app_py(module: str):
    sys.path.insert(0, str(ROOT / "services" / "telemetry-simulator"))
    # evict any foreign `app` so the telemetry-simulator package is unambiguous
    for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_mod, None)
    import importlib
    return importlib.import_module(module)


def test_service_wrapper_falls_back_without_artifact():
    cpd = _load_sim_app_py("app.engine.calibrated_soc_drop")
    svc = cpd.CalibratedSoCDropService(artifact_dir=ROOT / "nope")
    st = svc.status()
    assert st["active"] is False
    assert st["status"] == "physics_only_fallback"
    from vehicle_ai.schemas import CalibratedSoCDropRequest
    resp = svc.predict(CalibratedSoCDropRequest(physics_soc_drop_pct=3.0, speed_kph_mean=50.0))
    assert resp.status == "physics_only_fallback"
    assert resp.calibrated_soc_drop_pct == pytest.approx(3.0)


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_service_wrapper_active_path():
    cpd = _load_sim_app_py("app.engine.calibrated_soc_drop")
    svc = cpd.CalibratedSoCDropService(artifact_dir=ARTIFACT_DIR)
    st = svc.status()
    assert st["active"] is True
    assert st["model_version"] == "v2-ws1-r1"


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_endpoint_active_path():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    st = client.get("/api/prediction/status")
    assert st.status_code == 200
    assert st.json()["active"] is True

    req = {"speed_kph_mean": 75.0, "speed_kph_std": 8.0, "speed_kph_max": 90.0,
           "speed_kph_p90": 85.0, "distance_km": 6.0, "accel_mean_mps2": 0.1,
           "decel_fraction": 0.2, "regen_decel_energy_kwh": 0.05,
           "hv_aux_power_w_mean": 1300.0, "outside_temp_c_mean": 10.0,
           "coolant_temp_c_mean": 32.0, "battery_temp_inlet_c_mean": 25.0,
           "battery_temp_outlet_c_mean": 27.0, "motor_stator_temp_c_mean": 48.0,
           "motor_rotor_temp_c_mean": 53.0, "pack_voltage_v_mean": 349.0,
           "physics_soc_drop_pct": 2.8}
    r = client.post("/api/prediction/calibrated-soc-drop", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "active"
    assert body["speed_band"] == "60-90"
    assert body["lambda_applied"] == 1.0
    assert body["model_version"] == "v2-ws1-r1"
    # physics-only still embedded (reference)
    assert body["physics_soc_drop_pct"] == pytest.approx(2.8)