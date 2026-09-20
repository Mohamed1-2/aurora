"""WS2 calibrated SOC-drop prediction service (runtime integration).

Active prediction path (PROMOTED, frozen v2-ws1-r1):
    calibrated_soc_drop_pct = physics_soc_drop_pct + lam_band * ml_residual_pct

The calibration package lives at ``services/calibration`` — not importable at
service runtime (services/ has no __init__.py), mirroring how the TUM ingestor is
loaded. This module lazily:
  * adds ``services`` to sys.path and imports ``calibration.predict``;
  * builds a ``CalibratedSoCDropPredictor`` against the frozen artifact dir.

If LightGBM, the artifact, or the calibration package is unavailable the service
NEVER hard-fails: it returns a physics-only prediction with
``status="physics_only_fallback"`` and a human-readable reason. Physics-only is
therefore the runtime fallback/reference, deliberately retained.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vehicle_ai.schemas import (
    CalibratedSoCDropRequest,
    CalibratedSoCDropResponse,
    ExplainSoCDropRequest,
    PredictionExplanation,
)

# app/engine/calibrated_soc_drop.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVICES_DIR = _REPO_ROOT / "services"
_DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "calibration" / "v2-ws1"


def _load_predictor_cls():
    if str(_SERVICES_DIR) not in sys.path:
        sys.path.insert(0, str(_SERVICES_DIR))
    from calibration.predict import CalibratedSoCDropPredictor  # noqa: PLC0415

    return CalibratedSoCDropPredictor


def _load_explain_fn():
    if str(_SERVICES_DIR) not in sys.path:
        sys.path.insert(0, str(_SERVICES_DIR))
    from calibration.explain import build_explanation  # noqa: PLC0415

    return build_explanation


class CalibratedSoCDropService:
    """Lazy wrapper around the frozen calibration predictor with safe fallback."""

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir else _DEFAULT_ARTIFACT_DIR
        self._predictor = None
        self._load_error: str | None = None

    def _ensure_predictor(self):
        if self._predictor is not None or self._load_error is not None:
            return
        try:
            cls = _load_predictor_cls()
            self._predictor = cls(self.artifact_dir)
        except Exception as exc:  # noqa: BLE001 - service must never hard-fail
            self._predictor = None
            self._load_error = f"calibration service unavailable: {exc!r}"

    def status(self) -> dict:
        self._ensure_predictor()
        if self._predictor is None:
            return {
                "active": False,
                "status": "physics_only_fallback",
                "reason": self._load_error or "calibration service not loaded",
                "artifact_dir": str(self.artifact_dir),
            }
        meta = self._predictor.meta or {}
        return {
            "active": self._predictor.available,
            "status": "active" if self._predictor.available else "physics_only_fallback",
            "reason": self._predictor.error,
            "artifact_dir": str(self.artifact_dir),
            "model_version": self._predictor.version,
            "capacity_kwh": meta.get("capacity_kwh"),
            "features": meta.get("features"),
        }

    def predict(self, req: CalibratedSoCDropRequest) -> CalibratedSoCDropResponse:
        self._ensure_predictor()
        if self._predictor is None:
            return CalibratedSoCDropResponse(
                physics_soc_drop_pct=req.physics_soc_drop_pct,
                ml_residual_pct=0.0,
                calibrated_soc_drop_pct=req.physics_soc_drop_pct,
                speed_band="unknown",
                lambda_applied=0.0,
                model_version=None,
                status="physics_only_fallback",
                fallback_reason=self._load_error or "calibration service not loaded",
            )
        pred = self._predictor.predict(features=req.model_dump(), physics_soc_drop_pct=req.physics_soc_drop_pct)
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

    def explain(self, req: ExplainSoCDropRequest) -> PredictionExplanation:
        """Deterministic explanation (absolute or from→to delta) — never raises."""
        self._ensure_predictor()
        try:
            build_explanation = _load_explain_fn()
            return build_explanation(to=req.to, from_=req.from_, predictor=self._predictor)
        except Exception as exc:  # noqa: BLE001 - explainability must never hard-fail
            from vehicle_ai.schemas import CalibratedSoCDropResponse as _Resp  # noqa: PLC0415

            fallback = self.predict(req.to) if self._predictor is not None else CalibratedSoCDropResponse(
                physics_soc_drop_pct=req.to.physics_soc_drop_pct,
                ml_residual_pct=0.0,
                calibrated_soc_drop_pct=req.to.physics_soc_drop_pct,
                speed_band="unknown",
                lambda_applied=0.0,
                model_version=None,
                status="physics_only_fallback",
                fallback_reason=self._load_error or "calibration service not loaded",
            )
            return PredictionExplanation(
                shape="delta" if req.from_ is not None else "absolute",
                status="physics_only_fallback",
                fallback_reason=f"explanation unavailable: {exc!r}",
                model_version=fallback.model_version,
                capacity_kwh=(self._predictor.meta or {}).get("capacity_kwh") if self._predictor is not None else None,
                to_prediction=fallback,
                from_prediction=self.predict(req.from_) if self._predictor is not None and req.from_ is not None else None,
            )


calibrated_service = CalibratedSoCDropService()