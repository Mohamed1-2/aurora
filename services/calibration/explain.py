"""WS3 — deterministic explanation of the calibrated SOC-drop prediction.

The LLM is only a wording layer: every number and every ranked cause below is
computed in code and must be quoted verbatim, never re-derived by the model.

Two shapes:
  * absolute — why the ``to`` window's prediction is what it is;
  * delta    — why the prediction CHANGED between ``from`` and ``to``.

Attribution:
  * physics side — drivers reconstructed from window means (same WS1 constants
    as the calibration physics baseline), then NORMALIZED so the drivers sum
    exactly to the authoritative ``physics_soc_drop_pct`` (identity holds);
    hvac = 0 because TUM windows carry no cabin setpoint (never fabricated).
  * ML side — exact per-feature SHAP (``pred_contrib=True``) on the frozen
    model. Identity: ``sum(feature_contrib) + bias == residual`` to float
    precision. After lambda shrinkage a feature's *effective* contribution to
    ``calibrated_soc_drop_pct`` is ``λ * contrib_i`` and the base value is
    ``λ * bias`` — kept SEPARATE from user-facing feature causes (it is a model
    constant, never a physical cause).

All identity checks use a strict numeric tolerance, not float equality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vehicle_ai.schemas import (
    CalibratedSoCDropRequest,
    CalibratedSoCDropResponse,
    ContributionItem,
    PredictionExplanation,
    PredictionIdentityChecks,
)

from .physics import PhysicsBaseline
from .predict import _FEATURES, LAMBDA_POLICY_PCT, CalibratedSoCDropPredictor

_TOL_SOC_PTS = 1e-6
_WINDOW_S = 300.0
_TOP_CAUSES = 8

_DRIVERS = ("aero", "rolling", "inertia", "aux", "hvac")

_DRIVER_LABELS = {
    "aero": "Aerodynamic drag",
    "rolling": "Rolling resistance",
    "inertia": "Inertia / acceleration",
    "aux": "Auxiliary load",
    "hvac": "HVAC load",
}

_ML_LABELS = {
    "speed_kph_mean": "Average speed",
    "speed_kph_std": "Speed variability",
    "speed_kph_max": "Peak speed",
    "speed_kph_p90": "High-speed (p90) speed",
    "distance_km": "Distance covered",
    "accel_mean_mps2": "Mean acceleration",
    "decel_fraction": "Deceleration share",
    "regen_decel_energy_kwh": "Regenerative braking energy",
    "hv_aux_power_w_mean": "Auxiliary power draw",
    "outside_temp_c_mean": "Outside temperature",
    "coolant_temp_c_mean": "Coolant temperature",
    "battery_temp_inlet_c_mean": "Battery inlet temperature",
    "battery_temp_outlet_c_mean": "Battery outlet temperature",
    "motor_stator_temp_c_mean": "Motor stator temperature",
    "motor_rotor_temp_c_mean": "Motor rotor temperature",
    "pack_voltage_v_mean": "Pack voltage",
    "physics_soc_drop_pct": "Physics SOC drop (input)",
}


@dataclass(frozen=True)
class _WindowAttribution:
    """Everything derived for one window, in SOC points."""

    prediction: CalibratedSoCDropResponse
    physics_contribs: dict[str, float]  # driver key -> effective SOC pts (normalized, sums to physics)
    ml_contribs: dict[str, float]       # feature -> effective SOC pts = λ * SHAP_i
    ml_bias_effective: float            # λ * base value (model constant, separate)
    ml_residual: float                  # raw residual (sum SHAP + bias)
    capacity_kwh: float


def _reconstructed_driver_wh(req: CalibratedSoCDropRequest, bs: PhysicsBaseline) -> dict[str, float]:
    """Reconstruct per-driver energy (Wh) over the 300s window from feature means.

    Uses the same WS1 constants as the calibration physics baseline. This is a
    deterministic *attribution* reconstruction (window means), normalized
    downstream so the drivers always sum to the authoritative physics total.
    """
    v_ms = max(0.0, req.speed_kph_mean or 0.0) / 3.6
    accel = float(req.accel_mean_mps2 or 0.0)
    aux_power = max(0.0, float(req.hv_aux_power_w_mean or 0.0))
    wh = {
        "aero": 0.5 * bs.rho * bs.cd * bs.area * v_ms**3,
        "rolling": bs.mass_kg * bs.g * bs.crr * v_ms,
        "inertia": bs.mass_kg * accel * v_ms,
        "aux": aux_power,
        "hvac": 0.0,
    }
    return {k: max(0.0, wh[k]) * _WINDOW_S / 3600.0 for k in _DRIVERS}


def _normalize_drivers(raw_wh: dict[str, float], physics_soc_drop_pct: float) -> dict[str, float]:
    """Map reconstructed driver Wh to SOC points summed EXACTLY to the physics total."""
    total_raw = sum(raw_wh.values())
    if total_raw > 0.0:
        contribs = {k: physics_soc_drop_pct * (raw_wh[k] / total_raw) for k in _DRIVERS}
    else:
        # Degenerate (standstill / all-zero reconstruction): park the whole
        # physics total on aux with a note rather than break the identity.
        contribs = {k: (physics_soc_drop_pct if k == "aux" else 0.0) for k in _DRIVERS}
    return contribs


def _direction(value: float) -> str:
    if value > _TOL_SOC_PTS:
        return "increases_drop"
    if value < -_TOL_SOC_PTS:
        return "decreases_drop"
    return "neutral"


def _attribution(req: CalibratedSoCDropRequest, predictor: CalibratedSoCDropPredictor, bs: PhysicsBaseline) -> _WindowAttribution:
    features = req.model_dump()
    pred, raw_contribs, bias = predictor.predict_contrib(features, physics_soc_drop_pct=req.physics_soc_drop_pct)
    lam = pred.lambda_applied
    physics_contribs = _normalize_drivers(_reconstructed_driver_wh(req, bs), pred.physics_soc_drop_pct)

    if bias is None:
        ml_contribs = {k: 0.0 for k in _FEATURES}
        ml_bias_effective = None
    else:
        ml_contribs = {k: lam * raw_contribs[k] for k in _FEATURES}
        ml_bias_effective = lam * bias

    capacity = (predictor.meta or {}).get("capacity_kwh") or bs_capacity(bs)
    return _WindowAttribution(
        prediction=_response_from_pred_parts(pred),
        physics_contribs=physics_contribs,
        ml_contribs=ml_contribs,
        ml_bias_effective=ml_bias_effective,
        ml_residual=pred.ml_residual_pct,
        capacity_kwh=float(capacity),
    )


def _response_from_pred_parts(pred) -> CalibratedSoCDropResponse:
    return CalibratedSoCDropResponse(
        physics_soc_drop_pct=pred.physics_soc_drop_pct,
        ml_residual_pct=pred.ml_residual_pct,
        calibrated_soc_drop_pct=pred.calibrated_soc_drop_pct,
        speed_band=pred.band,
        lambda_applied=pred.lambda_applied,
        model_version=pred.model_version,
        status=pred.status,
        fallback_reason=pred.fallback_reason,
    )


def bs_capacity(bs: PhysicsBaseline) -> float:
    try:
        return float(bs.profile.value("battery_capacity_kwh"))
    except Exception:  # noqa: BLE001 - profile key may not exist
        return 58.0


def _physics_items(to: _WindowAttribution, from_att: _WindowAttribution | None) -> list[ContributionItem]:
    items = []
    for driver in _DRIVERS:
        val = to.physics_contribs[driver]
        delta = None
        if from_att is not None:
            delta = val - from_att.physics_contribs[driver]
        note = "normalized so drivers sum to physics_soc_drop_pct"
        if driver == "hvac":
            note = "hvac = 0: TUM windows carry no cabin setpoint (never fabricated)"
        items.append(
            ContributionItem(
                key=driver,
                label=_DRIVER_LABELS[driver],
                contribution_soc_pct=round(val, 5),
                delta_soc_pct=round(delta, 5) if delta is not None else None,
                value_wh=None,
                provenance="physics",
                direction=_direction(delta if delta is not None else val),
                description=note,
            )
        )
    return items


def _ml_items(to: _WindowAttribution, from_att: _WindowAttribution | None) -> list[ContributionItem]:
    items = []
    for feat in _FEATURES:
        val = to.ml_contribs[feat]
        delta = None
        if from_att is not None:
            delta = val - from_att.ml_contribs[feat]
        if val == 0.0 and (delta is None or delta == 0.0):
            continue  # skip no-op features to keep the block tight
        items.append(
            ContributionItem(
                key=feat,
                label=_ML_LABELS.get(feat, feat),
                contribution_soc_pct=round(val, 5),
                delta_soc_pct=round(delta, 5) if delta is not None else None,
                value_wh=None,
                provenance="ml",
                direction=_direction(delta if delta is not None else val),
                description="ML model attribution (SHAP on frozen v2-ws1-r1)",
            )
        )
    return items


def _top_causes(physics: list[ContributionItem], ml: list[ContributionItem]) -> list[ContributionItem]:
    combined = physics + ml
    key = lambda it: abs(it.delta_soc_pct) if it.delta_soc_pct is not None else abs(it.contribution_soc_pct)  # noqa: E731
    ranked = sorted(combined, key=key, reverse=True)
    return ranked[:_TOP_CAUSES]


def _identity(to: _WindowAttribution, from_att: _WindowAttribution | None) -> PredictionIdentityChecks:
    to_bias = to.ml_bias_effective or 0.0
    physics_sum = sum(to.physics_contribs.values())
    ml_sum = sum(to.ml_contribs.values()) + to_bias
    blend = to.prediction.physics_soc_drop_pct + to.prediction.lambda_applied * to.ml_residual
    delta_matches = None
    if from_att is not None:
        from_bias = from_att.ml_bias_effective or 0.0
        d_phys = sum(to.physics_contribs.values()) - sum(from_att.physics_contribs.values())
        d_ml = (sum(to.ml_contribs.values()) + to_bias) - (sum(from_att.ml_contribs.values()) + from_bias)
        d_calib = to.prediction.calibrated_soc_drop_pct - from_att.prediction.calibrated_soc_drop_pct
        delta_matches = math.isclose(d_phys + d_ml, d_calib, abs_tol=_TOL_SOC_PTS)
    return PredictionIdentityChecks(
        physics_sum_matches=math.isclose(physics_sum, to.prediction.physics_soc_drop_pct, abs_tol=_TOL_SOC_PTS),
        ml_sum_matches=math.isclose(ml_sum, to.prediction.calibrated_soc_drop_pct - to.prediction.physics_soc_drop_pct, abs_tol=_TOL_SOC_PTS),
        blend_matches=math.isclose(blend, to.prediction.calibrated_soc_drop_pct, abs_tol=_TOL_SOC_PTS),
        delta_matches=delta_matches,
        tolerance_soc_pct=_TOL_SOC_PTS,
    )


def build_explanation(
    to: CalibratedSoCDropRequest,
    from_: CalibratedSoCDropRequest | None = None,
    predictor: CalibratedSoCDropPredictor | None = None,
) -> PredictionExplanation:
    """Deterministic explanation of the ``to`` prediction (absolute) or the
    ``from`` + ``to`` change (delta/what-if). Never raises.
    """
    from ._bootstrap import repo_root  # noqa: PLC0415

    if predictor is None:
        predictor = CalibratedSoCDropPredictor(repo_root() / "artifacts" / "calibration" / "v2-ws1")
    bs = PhysicsBaseline()

    to_att = _attribution(to, predictor, bs)
    from_att = _attribution(from_, predictor, bs) if from_ is not None else None

    phys_items = _physics_items(to_att, from_att)
    ml_items = _ml_items(to_att, from_att)

    return PredictionExplanation(
        shape="delta" if from_ is not None else "absolute",
        status=to_att.prediction.status,
        fallback_reason=to_att.prediction.fallback_reason,
        model_version=to_att.prediction.model_version,
        capacity_kwh=to_att.capacity_kwh,
        to_prediction=to_att.prediction,
        from_prediction=from_att.prediction if from_att is not None else None,
        physics_contributions=phys_items,
        ml_contributions=ml_items,
        ml_bias_soc_pct=round(to_att.ml_bias_effective, 5) if to_att.ml_bias_effective is not None else None,
        top_causes=_top_causes(phys_items, ml_items),
        identity_checks=_identity(to_att, from_att),
    )