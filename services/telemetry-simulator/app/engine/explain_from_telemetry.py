"""Runtime bridge: raw telemetry frames -> WS3 deterministic explanation.

The assistant holds a rolling buffer of *live* vehicle frames (or a replay
snapshot). Instead of re-deriving the canonical feature vector itself, it posts
those frames here; this module bins them into 10 s bins, runs the same
``calibration.features.extract_features`` + WS1 physics baseline used for the
frozen model, then ``build_explanation``. Every number stays computed in the
calibration stack — the assistant only forwards telemetry and styles the block.

Never raises: any failure falls back to a physics-only explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from vehicle_ai.schemas import CalibratedSoCDropRequest, PredictionExplanation

_BIN_S = 10.0
_DEFAULT_HZ = 10.0
_DEFAULT_CAPACITY_KWH = 58.0
_MIN_BINS = 3  # 30 s minimum for a meaningful physics attribution

# app/engine/explain_from_telemetry.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVICES_DIR = _REPO_ROOT / "services"


def _as_number(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return fmean(vals)


def frames_to_bins(
    frames: list[dict], hz: float = _DEFAULT_HZ, bin_s: float = _BIN_S
) -> list[dict]:
    """Aggregate raw frames into 10 s bins in the ``extract_features`` schema.

    Live simulator signals are mapped to training-bin keys; thermal signals the
    simulator does not emit (pack in/outlet, stator/rotor) are left absent,
    which ``extract_features`` reports as None/NaN (handled by the frozen
    predictor). A frame is either ``{"signals": {...}, ...}`` or a bare signals
    dict (the replay-snapshot shape).
    """
    dt = 1.0 / hz
    per_bin = max(1, int(round(bin_s / dt)))
    bins: list[dict] = []
    for start in range(0, len(frames), per_bin):
        chunk = frames[start : start + per_bin]
        signals = [f.get("signals", f) for f in chunk]
        speed = [_as_number(s.get("vehicle_speed_kph")) for s in signals]
        hvac_kw = [_as_number(s.get("hvac_power_kw")) for s in signals]

        def _aux_power_w() -> float | None:
            # hvac energy is logged per tick in Wh -> W = wh / dt * 3600
            hvac_wh = [_as_number(s.get("hvac_energy_wh_tick")) for s in signals]
            aux_wh = [_as_number(s.get("aux_energy_wh_tick")) for s in signals]
            hvac_w = _mean([(v / dt) * 3600.0 if v is not None else None for v in hvac_wh])
            aux_w = _mean([(v / dt) * 3600.0 if v is not None else None for v in aux_wh])
            if hvac_w is None and aux_w is None and hvac_kw:
                return _mean([v * 1000.0 if v is not None else None for v in hvac_kw])
            return (hvac_w or 0.0) + (aux_w or 0.0)

        bins.append(
            {
                "speed_kph": _mean(speed) or 0.0,
                "hv_aux_power_w": _aux_power_w(),
                "outside_temp_c": _mean([_as_number(s.get("outside_temp_c")) for s in signals]),
                "coolant_temp_c": _mean([_as_number(s.get("coolant_temp_c")) for s in signals]),
                "pack_voltage_v": _mean([_as_number(s.get("battery_voltage_v")) for s in signals]),
            }
        )
    return bins


def _window_meta(frames: list[dict], bins: list[dict]) -> tuple[float, float, float, list[float]]:
    first = frames[0].get("signals", frames[0])
    last = frames[-1].get("signals", frames[-1])
    soc_start = _as_number(first.get("battery_soc_pct"))
    soc_end = _as_number(last.get("battery_soc_pct"))
    speeds = [(b.get("speed_kph", 0.0) or 0.0) for b in bins]
    moving = sum(1 for v in speeds if v > 0.0) / max(1, len(speeds))
    return soc_start, soc_end, moving, speeds


def features_from_frames(
    frames: list[dict], hz: float = _DEFAULT_HZ, capacity_kwh: float = _DEFAULT_CAPACITY_KWH
) -> CalibratedSoCDropRequest | None:
    """Convert live frames -> canonical feature vector (or None if too short)."""
    if not frames:
        return None
    bins = frames_to_bins(frames, hz=hz)
    if len(bins) < _MIN_BINS:
        return None

    if str(_SERVICES_DIR) not in sys.path:
        sys.path.insert(0, str(_SERVICES_DIR))
    from calibration.features import extract_features  # noqa: PLC0415
    from calibration.physics import PhysicsBaseline  # noqa: PLC0415
    from calibration.windowing import RawWindow  # noqa: PLC0415

    soc_start, soc_end, moving, _ = _window_meta(frames, bins)
    window = RawWindow(
        vehicle_id=next((f.get("vehicle_id") for f in frames if f.get("vehicle_id")), "orion-x"),
        episode_id="live-runtime",
        run_id="live-runtime",
        split="runtime",
        window_dur_s=len(bins) * _BIN_S,
        moving_fraction=round(moving, 4),
        soc_start_pct=soc_start or 0.0,
        soc_end_pct=soc_end or 0.0,
        observed_soc_drop_pct=round((soc_start or 0.0) - (soc_end or 0.0), 4),
        n_bins=len(bins),
        bins=bins,
    )

    physics = PhysicsBaseline()
    row = extract_features(window, physics, capacity_kwh)
    keys = CalibratedSoCDropRequest.model_fields.keys()
    return CalibratedSoCDropRequest.model_validate({k: row.get(k) for k in keys})


def explain_from_frames(
    to_frames: list[dict],
    from_frames: list[dict] | None = None,
    hz: float = _DEFAULT_HZ,
    capacity_kwh: float = _DEFAULT_CAPACITY_KWH,
) -> PredictionExplanation:
    """Deterministic explanation from raw telemetry frames (delta when both given)."""
    from vehicle_ai.schemas import CalibratedSoCDropResponse as _Resp  # noqa: PLC0415

    def _fallback(msg: str, to_req: CalibratedSoCDropRequest | None) -> PredictionExplanation:
        drop = to_req.physics_soc_drop_pct if to_req is not None else 0.0
        pred = _Resp(
            physics_soc_drop_pct=drop,
            ml_residual_pct=0.0,
            calibrated_soc_drop_pct=drop,
            speed_band="unknown",
            lambda_applied=0.0,
            model_version=None,
            status="physics_only_fallback",
            fallback_reason=msg,
        )
        return PredictionExplanation(
            shape="delta" if from_frames is not None else "absolute",
            status="physics_only_fallback",
            fallback_reason=msg,
            to_prediction=pred,
            from_prediction=None,
            capacity_kwh=capacity_kwh,
        )

    to_req = features_from_frames(to_frames, hz=hz, capacity_kwh=capacity_kwh)
    if to_req is None:
        return _fallback("insufficient telemetry to build a prediction window", None)

    from_req = None
    if from_frames:
        from_req = features_from_frames(from_frames, hz=hz, capacity_kwh=capacity_kwh)

    if str(_SERVICES_DIR) not in sys.path:
        sys.path.insert(0, str(_SERVICES_DIR))
    from calibration.explain import build_explanation  # noqa: PLC0415

    try:
        return build_explanation(to=to_req, from_=from_req)
    except Exception as exc:  # noqa: BLE001 - explainability must never hard-fail
        return _fallback(f"explanation unavailable: {exc!r}", to_req)