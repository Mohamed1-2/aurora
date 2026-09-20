"""Availability report for the WS2 training dataset.

Reports, after windowing+filtering: usable windows by vehicle and split, and the
missing-value rate of every proposed feature (fraction of rows where the feature
aggregate is NaN). Optional thermal signals are reported separately so reviewers
can see exactly what the model can and cannot rely on.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


FEATURE_KEYS = [
    "speed_kph_mean", "speed_kph_std", "speed_kph_max", "speed_kph_p90",
    "distance_km", "accel_mean_mps2", "decel_fraction", "regen_decel_energy_kwh",
    "hv_aux_power_w_mean", "outside_temp_c_mean", "coolant_temp_c_mean",
    "battery_temp_inlet_c_mean", "battery_temp_outlet_c_mean",
    "motor_stator_temp_c_mean", "motor_rotor_temp_c_mean", "pack_voltage_v_mean",
]


def missingness(rows: list[dict], feature_keys: list[str] = FEATURE_KEYS) -> dict[str, float]:
    n = max(1, len(rows))
    return {k: round(sum(1 for r in rows if r.get(k) is None) / n, 4) for k in feature_keys}


def build_availability(rows: list[dict], bin_counts: dict[str, int]) -> dict:
    by_vehicle: dict[str, list] = {}
    for r in rows:
        by_vehicle.setdefault(r["vehicle_id"], []).append(r)
    split_counts = Counter(r["split"] for r in rows)
    return {
        "input_10s_bins_by_vehicle": bin_counts,
        "usable_300s_windows": {
            "total": len(rows),
            "by_vehicle": {v: len(rows_v) for v, rows_v in sorted(by_vehicle.items())},
            "by_split": dict(sorted(split_counts.items())),
        },
        "feature_missing_fraction": missingness(rows),
        "note": ("windows are dense-5-core gated (soc, speed, aux, ambient, coolant) "
                 "+ moving_fraction>=0.8 + net-discharge; optional thermal signals "
                 "may still be NaN (feature_missing_fraction)."),
    }


def write_availability(out_dir: Path, report: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "availability_report.json"
    with p.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p