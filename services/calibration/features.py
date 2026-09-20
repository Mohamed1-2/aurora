"""Feature extraction for 300 s real-TUM training windows.

All features come from measured (or derived-from-measured) telemetry. Optional
thermal signals (battery pack in/outlet, stator/rotor, hv_temp_max) are NaN
when < 50 % of the bins in the window carry them; the missingness is reported
as a first-class number in the availability report.

Convention: ``soc_drop_*``/``residual_pct`` are positive-while-discharging.
"""

from __future__ import annotations

from statistics import mean, pstdev

from .physics import PhysicsBaseline
from .windowing import RawWindow

_SCALE = 1.0 / 3.6


def _mean_of(bins: list[dict], key: str, default: float | None = None) -> float | None:
    vals = [b[key] for b in bins if key in b and b[key] is not None]
    if not vals:
        return default
    return mean(vals)


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = 0.9 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def extract_features(window: RawWindow, physics: PhysicsBaseline, capacity_kwh: float) -> dict:
    bins = window.bins
    speeds = [b.get("speed_kph", 0.0) for b in bins]
    n = len(speeds)
    accels: list[float] = []
    decel = 0
    regen_kwh = 0.0
    v_prev = None
    for v in speeds:
        vms = v * _SCALE
        if v_prev is not None:
            a = (vms - v_prev) / 10.0
            accels.append(a)
            if a < 0:
                decel += 1
            if vms < v_prev:
                regen_kwh += 0.5 * physics.mass_kg * (v_prev * v_prev - vms * vms) / 3.6e6
        v_prev = vms

    distance_km = sum(speeds) * _SCALE * 10.0 / 1000.0
    aux_mean = _mean_of(bins, "hv_aux_power_w", 0.0)
    aux_wh = physics.aux_wh(bins)
    motive_wh = physics.motive_wh(bins)

    physics_drop = physics.soc_drop_phys_pct(bins, capacity_kwh)

    features = {
        "vehicle_id": window.vehicle_id,
        "episode_id": window.episode_id,
        "run_id": window.run_id,
        "split": window.split,
        "window_dur_s": window.window_dur_s,
        "n_bins": window.n_bins,
        "moving_fraction": window.moving_fraction,
        # window metadata
        "start_utc": window.start_utc,
        "end_utc": window.end_utc,
        "soc_start_pct": window.soc_start_pct,
        "soc_end_pct": window.soc_end_pct,
        "quant_steps": window.quant_steps,
        # measured / derived features
        "speed_kph_mean": round(mean(speeds), 3),
        "speed_kph_std": round(pstdev(speeds), 3),
        "speed_kph_max": round(max(speeds), 3),
        "speed_kph_p90": round(_p90(speeds), 3),
        "distance_km": round(distance_km, 4),
        "accel_mean_mps2": round(mean(accels), 4) if accels else 0.0,
        "decel_fraction": round(decel / max(1, len(accels)), 4),
        "regen_decel_energy_kwh": round(regen_kwh, 4),
        "hv_aux_power_w_mean": round(aux_mean, 3) if aux_mean is not None else None,
        "outside_temp_c_mean": _mean_of(bins, "outside_temp_c"),
        "coolant_temp_c_mean": _mean_of(bins, "coolant_temp_c"),
        "battery_temp_inlet_c_mean": _mean_of(bins, "battery_temp_inlet_c"),
        "battery_temp_outlet_c_mean": _mean_of(bins, "battery_temp_outlet_c"),
        "motor_stator_temp_c_mean": _mean_of(bins, "motor_stator_temp_c"),
        "motor_rotor_temp_c_mean": _mean_of(bins, "motor_rotor_temp_c"),
        "pack_voltage_v_mean": _mean_of(bins, "pack_voltage_v"),
        # physics / target (positive-drop convention)
        "physics_motive_wh": round(motive_wh, 3),
        "physics_aux_wh": round(aux_wh, 3),
        "physics_soc_drop_pct": physics_drop,
        "observed_soc_drop_pct": window.observed_soc_drop_pct,
        "residual_pct": round(window.observed_soc_drop_pct - physics_drop, 4),
        "capacity_kwh": capacity_kwh,
    }
    return features


def missingness(features_rows: list[dict], feature_keys: list[str]) -> dict:
    """Fraction (0..1) of rows where each feature is missing (None)."""
    out: dict[str, float] = {}
    for key in feature_keys:
        n_missing = sum(1 for r in features_rows if r.get(key) is None)
        out[key] = round(n_missing / max(1, len(features_rows)), 4)
    return out