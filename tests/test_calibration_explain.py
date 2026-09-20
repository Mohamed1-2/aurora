"""WS3 explainability tests (deterministic SOC-drop explanations).

Contracts covered:
* Identity guarantees — physics sum, ML (SHAP) sum, blend, and delta must
  hold within the strict numeric tolerance (never literal float equality).
* Fallback — artifact unavailable => physics-only explanation, never raises.
* Prompt v8 — the ``## PREDICTION EXPLANATION`` block is injected ONLY when the
  ``why_prediction`` intent is active AND an explanation is supplied; numbers
  are quoted verbatim and ML attributions are never causality.
* Real two-window regression from ``dataset_w300s.parquet`` — physics driver
  normalization + SHAP attribution on actual windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "services"
for _p in (str(SERVICES), str(ROOT / "services" / "assistant-api")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

ARTIFACT_DIR = ROOT / "artifacts" / "calibration" / "v2-ws1"
HAS_ARTIFACT = ARTIFACT_DIR.exists() and (ARTIFACT_DIR / "calibration_model.lgb").exists()

from calibration.explain import build_explanation  # noqa: E402
from calibration.predict import CalibratedSoCDropPredictor  # noqa: E402
from vehicle_ai.schemas import CalibratedSoCDropRequest  # noqa: E402


def _features(speed_kph_mean: float = 80.0, physics: float = 2.5, **overrides) -> CalibratedSoCDropRequest:
    f = {
        "speed_kph_mean": speed_kph_mean,
        "speed_kph_std": 6.0,
        "speed_kph_max": speed_kph_mean + 15.0,
        "speed_kph_p90": speed_kph_mean + 8.0,
        "distance_km": speed_kph_mean * 300.0 / 3600.0,
        "accel_mean_mps2": 0.05,
        "decel_fraction": 0.2,
        "regen_decel_energy_kwh": 0.05,
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
    return CalibratedSoCDropRequest.model_validate(f)


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_absolute_explanation_identities_hold_within_tolerance():
    e = build_explanation(_features(physics=2.5))
    chk = e.identity_checks
    assert e.shape == "absolute"
    assert chk.physics_sum_matches is True
    assert chk.ml_sum_matches is True
    assert chk.blend_matches is True
    assert chk.delta_matches is None
    assert chk.tolerance_soc_pct == 1e-6
    # bias is a separate field, never a feature cause
    assert e.ml_bias_soc_pct is not None
    assert all(c.key != "ml_bias_soc_pct" for c in e.ml_contributions)


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_delta_explanation_identities_hold():
    to = _features(speed_kph_mean=100.0, physics=3.1)
    fro = _features(speed_kph_mean=60.0, physics=1.8)
    e = build_explanation(to, from_=fro)
    chk = e.identity_checks
    assert e.shape == "delta"
    assert chk.physics_sum_matches is True
    assert chk.ml_sum_matches is True
    assert chk.blend_matches is True
    assert chk.delta_matches is True
    assert e.from_prediction is not None
    # delta causes are populated per item
    d = [c for c in e.top_causes if c.delta_soc_pct is not None]
    assert len(d) > 0


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_physics_drivers_normalized_to_sum():
    e = build_explanation(_features(physics=2.5))
    phys_sum = sum(c.contribution_soc_pct for c in e.physics_contributions)
    # 5-dp display rounding of 5 items -> allow ~2.5e-5 drift
    assert phys_sum == pytest.approx(2.5, abs=5e-5)
    keys = {c.key for c in e.physics_contributions}
    assert {"aero", "rolling", "inertia", "aux", "hvac"} <= keys
    hvac = next(c for c in e.physics_contributions if c.key == "hvac")
    assert hvac.contribution_soc_pct == 0.0  # no cabin setpoint -> never fabricated


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_ml_provenance_is_correlation_not_causality():
    e = build_explanation(_features(physics=2.5))
    for c in e.ml_contributions:
        assert c.provenance == "ml"
        assert "ML model attribution" in c.description
    # physics driver causals keep their physical meaning
    aero = next(c for c in e.physics_contributions if c.key == "aero")
    assert aero.provenance == "physics"


# --- fallback -----------------------------------------------------------------


def test_explain_fallback_physics_only_never_raises():
    e = build_explanation(
        _features(physics=2.0),
        predictor=CalibratedSoCDropPredictor(ROOT / "artifacts" / "calibration" / "does-not-exist"),
    )
    assert e.status == "physics_only_fallback"
    assert e.fallback_reason
    assert e.to_prediction.calibrated_soc_drop_pct == pytest.approx(2.0)
    assert e.ml_bias_soc_pct is None
    assert e.ml_contributions == []


# --- prompt v8 gating ---------------------------------------------------------


def _load_assistant_py(module: str):
    sys.path.insert(0, str(ROOT / "services" / "assistant-api"))
    for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_mod, None)
    import importlib

    return importlib.import_module(module)


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_prompt_v8_block_injected_only_when_intent_and_explanation():
    intents = _load_assistant_py("app.core.intents")
    prompt = _load_assistant_py("app.core.prompt")
    from vehicle_ai.schemas import Grounding

    e = build_explanation(_features(physics=2.2))
    why = "Why did my SOC drop?"
    assert "why_prediction" in intents.detect(why)

    msgs = prompt.build_messages(why, Grounding(), intents.detect(why), explanation=e)
    assert "## PREDICTION EXPLANATION" in msgs[1].content
    assert "Top causes (ranked)" in msgs[1].content

    # intent present, no explanation -> no block
    msgs_none = prompt.build_messages(why, Grounding(), intents.detect(why))
    assert "## PREDICTION EXPLANATION" not in msgs_none[1].content

    # explanation present, intent absent -> no block
    no_why = "How far can I go today?"
    assert "why_prediction" not in intents.detect(no_why)
    msgs_nointent = prompt.build_messages(no_why, Grounding(), intents.detect(no_why), explanation=e)
    assert "## PREDICTION EXPLANATION" not in msgs_nointent[1].content


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_prompt_v8_block_quotes_numbers_verbatim():
    """The block's numbers must come from the deterministic explanation only."""
    intents = _load_assistant_py("app.core.intents")
    prompt = _load_assistant_py("app.core.prompt")
    from vehicle_ai.schemas import Grounding

    e = build_explanation(_features(physics=2.5))
    msgs = prompt.build_messages("Why did my SOC drop?", Grounding(), ["why_prediction"], explanation=e)
    block = msgs[1].content.split("## PREDICTION EXPLANATION")[1]
    top = e.top_causes[0]
    fmt_label = top.label  # deterministic label must appear verbatim
    assert fmt_label in block
    # no fabricated driver energy quantities (only capacity kWh is quoted)
    assert " Wh" not in block
    assert "value_wh" not in block


# --- real two-window regression -----------------------------------------------


@pytest.mark.skipif(not HAS_ARTIFACT, reason="frozen v2-ws1-r1 artifact not present")
def test_real_windows_from_dataset_two_window_regression():
    import pandas as pd

    df = pd.read_parquet(ARTIFACT_DIR / "dataset_w300s.parquet")
    assert len(df) >= 2
    cols = {
        "speed_kph_mean": "speed_kph_mean", "speed_kph_std": "speed_kph_std",
        "speed_kph_max": "speed_kph_max", "speed_kph_p90": "speed_kph_p90",
        "distance_km": "distance_km", "accel_mean_mps2": "accel_mean_mps2",
        "decel_fraction": "decel_fraction", "regen_decel_energy_kwh": "regen_decel_energy_kwh",
        "hv_aux_power_w_mean": "hv_aux_power_w_mean", "outside_temp_c_mean": "outside_temp_c_mean",
        "coolant_temp_c_mean": "coolant_temp_c_mean",
        "battery_temp_inlet_c_mean": "battery_temp_inlet_c_mean",
        "battery_temp_outlet_c_mean": "battery_temp_outlet_c_mean",
        "motor_stator_temp_c_mean": "motor_stator_temp_c_mean",
        "motor_rotor_temp_c_mean": "motor_rotor_temp_c_mean",
        "pack_voltage_v_mean": "pack_voltage_v_mean",
        "physics_soc_drop_pct": "physics_soc_drop_pct",
    }
    r0, r1 = df.iloc[0].to_dict(), df.iloc[1].to_dict()
    to = {k: (float(r1[c]) if not pd.isna(r1[c]) else None) for k, c in cols.items()}
    fro = {k: (float(r0[c]) if not pd.isna(r0[c]) else None) for k, c in cols.items()}
    e = build_explanation(CalibratedSoCDropRequest.model_validate(to),
                          from_=CalibratedSoCDropRequest.model_validate(fro))
    assert e.shape == "delta"
    assert e.identity_checks.physics_sum_matches is True
    assert e.identity_checks.ml_sum_matches is True
    assert e.identity_checks.delta_matches is True