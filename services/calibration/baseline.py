"""Physics-only baseline metrics on the held-out val/test splits.

Reports SOC-point error (primary, measured) in the locked positive-drop
convention plus a capacity-sensitivity analysis: the same underlying energy is
converted to ``soc_drop_pct`` with the per-vehicle flagged capacity across a
plausible band (55 / 58 / 62 / 82 kWh) so reviewers can see how sensitive the
physics baseline is to the capacity assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

SENSITIVITY_CAPACITY_KWH = (55.0, 58.0, 62.0, 82.0)


def _mae(obs, pred) -> float:
    return round(sum(abs(o - p) for o, p in zip(obs, pred)) / max(1, len(obs)), 4)


def _rmse(obs, pred) -> float:
    n = max(1, len(obs))
    return round((sum((o - p) ** 2 for o, p in zip(obs, pred)) / n) ** 0.5, 4)


def _mean_error(obs, pred) -> float:
    return round(sum(o - p for o, p in zip(obs, pred)) / max(1, len(obs)), 4)


def _speed_band(speed_mean: float) -> str:
    if speed_mean < 30: return "0-30"
    if speed_mean < 60: return "30-60"
    if speed_mean < 90: return "60-90"
    if speed_mean < 120: return "90-120"
    return "120+"


def _temp_band(temp_c: float | None) -> str:
    if temp_c is None: return "missing"
    if temp_c < 0: return "<0"
    if temp_c < 10: return "0-10"
    if temp_c < 20: return "10-20"
    if temp_c < 30: return "20-30"
    return "30+"


def _metrics(rows: list[dict], obs_key: str, pred: list[float]) -> dict:
    obs = [r[obs_key] for r in rows]
    return {
        "n": len(rows),
        "mae_pct": _mae(obs, pred),
        "rmse_pct": _rmse(obs, pred),
        "bias_pct": _mean_error(obs, pred),
    }


def compute_baseline(rows: list[dict]) -> dict:
    result: dict = {}
    for split in ("val", "test"):
        sub = [r for r in rows if r["split"] == split] or []
        result[split] = {"overall": _metrics(sub, "observed_soc_drop_pct",
                                             _physics_preds(sub))}
        by_vehicle: dict[str, list] = {}
        by_speed: dict[str, list] = {}
        by_temp: dict[str, list] = {}
        for r in sub:
            by_vehicle.setdefault(r["vehicle_id"], []).append(r)
            by_speed.setdefault(_speed_band(r["speed_kph_mean"]), []).append(r)
            by_temp.setdefault(_temp_band(r["outside_temp_c_mean"]), []).append(r)
        result[split]["by_vehicle"] = {veh: _metrics(v, "observed_soc_drop_pct", _physics_preds(v))
                                       for veh, v in sorted(by_vehicle.items())}
        result[split]["by_speed_band"] = {b: _metrics(v, "observed_soc_drop_pct", _physics_preds(v))
                                          for b, v in sorted(by_speed.items())}
        result[split]["by_temp_band"] = {b: _metrics(v, "observed_soc_drop_pct", _physics_preds(v))
                                         for b, v in sorted(by_temp.items())}

    def power_wh(r: dict) -> float:
        return r.get("physics_motive_wh", 0.0) + r.get("physics_aux_wh", 0.0)

    result["capacity_sensitivity_pct_soc_drop"] = {
        cap: {
            split: _metrics([r for r in rows if r["split"] == split], "observed_soc_drop_pct",
                            _pred_list(rows, split, cap))
            for split in ("val", "test")
        }
        for cap in SENSITIVITY_CAPACITY_KWH
    }
    result["convention"] = ("soc_drop_pct = soc_start_pct - soc_end_pct (positive while "
                            "discharging); residual_pct = observed - physics")
    result["note"] = ("soc-point error is primary (measured). energy/kWh error is "
                      "capacity-dependent and not reported directly here.")
    return result


def _physics_preds(rows: list[dict]) -> list[float]:
    return [r["physics_soc_drop_pct"] for r in rows]


def _pred_list(rows: list[dict], split: str, capacity_kwh: float) -> list[float]:
    return [round((r.get("physics_motive_wh", 0.0) + r.get("physics_aux_wh", 0.0))
                  / (capacity_kwh * 1000.0) * 100.0, 4)
            for r in rows if r["split"] == split]


def write_baseline(base_dir: Path, result: dict) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    p = base_dir / "baseline_val_test_metrics.json"
    with p.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p