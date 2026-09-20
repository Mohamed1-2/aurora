"""Runtime prediction interface for the PROMOTED conservative calibration (WS2).

Active prediction path (frozen, v2-ws1-r1):
    calibrated_soc_drop_pct = physics_soc_drop_pct + lam_band * ml_residual_pct
where ``lam_band`` is the per-speed-band shrinkage tuned on the validation split
only (0-30: 0.20 | 30-60: 1.00 | 60-90: 1.00 | 90-120: 1.00 | 120+: 0.55).

This module is the CLEAN interface: ``CalibratedSoCDropPredictor`` lazily loads
the frozen LightGBM artifact + metadata and never raises on missing models — it
degrades to a physics-only prediction with an explicit ``status`` and reason so
callers can always get an answer even when the ML artifact is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LAMBDA_POLICY_PCT = {"0-30": 0.20, "30-60": 1.00, "60-90": 1.00, "90-120": 1.00, "120+": 0.55}

BANDS = ["0-30", "30-60", "60-90", "90-120", "120+"]

MODEL_FILE = "calibration_model.lgb"
MODEL_META_FILE = "model_meta.json"

_FEATURES = [
    "speed_kph_mean", "speed_kph_std", "speed_kph_max", "speed_kph_p90",
    "distance_km", "accel_mean_mps2", "decel_fraction", "regen_decel_energy_kwh",
    "hv_aux_power_w_mean", "outside_temp_c_mean", "coolant_temp_c_mean",
    "battery_temp_inlet_c_mean", "battery_temp_outlet_c_mean",
    "motor_stator_temp_c_mean", "motor_rotor_temp_c_mean", "pack_voltage_v_mean",
    "physics_soc_drop_pct",
]


@dataclass(frozen=True)
class CalibrationPrediction:
    physics_soc_drop_pct: float
    ml_residual_pct: float
    calibrated_soc_drop_pct: float
    band: str
    lambda_applied: float
    model_version: str | None
    status: str  # "active" | "physics_only_fallback"
    fallback_reason: str = ""


def _band_of(speed_kph_mean: float) -> str:
    if speed_kph_mean < 30:
        return "0-30"
    if speed_kph_mean < 60:
        return "30-60"
    if speed_kph_mean < 90:
        return "60-90"
    if speed_kph_mean < 120:
        return "90-120"
    return "120+"


class CalibratedSoCDropPredictor:
    """Prediction interface backed by the frozen model (or physics fallback).

    Loading is lazy, so importing this module never requires LightGBM or the
    artifact to exist. ``available`` is False when the artifact or LightGBM is
    missing; ``predict`` then returns physics directly with status
    ``physics_only_fallback``.
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else None
        self._model = None
        self._meta: dict | None = None
        self._load_error: str | None = None

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        if self.artifact_dir is None:
            self._load_error = "no artifact dir provided"
            return
        model_path = self.artifact_dir / MODEL_FILE
        meta_path = self.artifact_dir / MODEL_META_FILE
        if not model_path.exists():
            self._load_error = f"model file missing: {model_path}"
            return
        if not meta_path.exists():
            self._load_error = f"model meta missing: {meta_path}"
            return
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(model_path))
            with meta_path.open() as fh:
                self._meta = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - runtime must never hard-fail
            self._model = None
            self._meta = None
            self._load_error = f"model load failed: {exc!r}"

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None and self._meta is not None

    @property
    def version(self) -> str | None:
        self._load()
        return self._meta.get("version") if self._meta else None

    @property
    def meta(self) -> dict | None:
        self._load()
        return self._meta

    @property
    def error(self) -> str | None:
        self._load()
        return self._load_error

    # -- prediction ---------------------------------------------------------

    def _row(self, features: dict, physics_soc_drop_pct: float) -> "list[float]":
        row = []
        for key in _FEATURES:
            if key == "physics_soc_drop_pct":
                row.append(float(physics_soc_drop_pct))
            else:
                v = features.get(key)
                row.append(float(v) if v is not None else float("nan"))
        return row

    def predict(self, features: dict, physics_soc_drop_pct: float | None = None) -> CalibrationPrediction:
        """Return the calibrated SOC-drop for a 300s window feature vector.

        ``features`` may contain all ``_FEATURES`` keys; missing keys and None
        values are treated as NaN (unknown), except ``physics_soc_drop_pct``
        which defaults to the explicit argument (or 0.0 if neither given).
        """
        if physics_soc_drop_pct is None:
            physics_soc_drop_pct = float(features.get("physics_soc_drop_pct", 0.0) or 0.0)
        band = _band_of(float(features.get("speed_kph_mean", 0.0) or 0.0))
        lam = LAMBDA_POLICY_PCT[band]

        self._load()
        if self._model is None:
            return CalibrationPrediction(
                physics_soc_drop_pct=physics_soc_drop_pct,
                ml_residual_pct=0.0,
                calibrated_soc_drop_pct=physics_soc_drop_pct,
                band=band,
                lambda_applied=0.0,
                model_version=None,
                status="physics_only_fallback",
                fallback_reason=self._load_error or "model unavailable",
            )

        import numpy as np
        X = np.array([self._row(features, physics_soc_drop_pct)], dtype=float)
        n_iters = self._model.num_trees()
        residual = float(self._model.predict(X, num_iteration=n_iters)[0])
        calibrated = physics_soc_drop_pct + lam * residual
        return CalibrationPrediction(
            physics_soc_drop_pct=physics_soc_drop_pct,
            ml_residual_pct=residual,
            calibrated_soc_drop_pct=calibrated,
            band=band,
            lambda_applied=lam,
            model_version=self.version,
            status="active",
        )

    def predict_contrib(self, features: dict, physics_soc_drop_pct: float | None = None):
        """Per-feature SHAP attribution of the ML residual (WS3 explainability).

        Returns ``(prediction, feature_contribs, bias)`` where
        ``feature_contribs`` is a ``dict[_FEATURES -> float]`` aligned in model
        feature order and ``bias`` is the LightGBM base (bias) value. Identity:
        ``sum(feature_contribs.values()) + bias == ml_residual_pct`` exactly
        (to float precision). On fallback returns ``(prediction, {}, None)``.
        """
        if physics_soc_drop_pct is None:
            physics_soc_drop_pct = float(features.get("physics_soc_drop_pct", 0.0) or 0.0)
        pred = self.predict(features, physics_soc_drop_pct)
        import numpy as np
        self._load()
        if self._model is None:
            return pred, {}, None
        X = np.array([self._row(features, physics_soc_drop_pct)], dtype=float)
        n_iters = self._model.num_trees()
        contrib = self._model.predict(X, num_iteration=n_iters, pred_contrib=True)[0]
        feature_contribs = {name: float(contrib[i]) for i, name in enumerate(_FEATURES)}
        bias = float(contrib[-1])
        return pred, feature_contribs, bias