"""WS2 held-out evaluation: physics-only vs physics+calibration.

Primary metric is SOC-point error (measured), in the locked positive-drop
convention. Reports MAE/RMSE/bias overall, by vehicle, by speed band, by
temperature band, the regression analysis (where ML worsens physics), and the
promotion decision per the approved rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import FEATURES, predict_delta_soc_row


def _speed_band(speed_mean: float) -> str:
    if speed_mean < 30: return "0-30"
    if speed_mean < 60: return "30-60"
    if speed_mean < 90: return "60-90"
    if speed_mean < 120: return "90-120"
    return "120+"


def _temp_band(t: float | None) -> str:
    if t is None: return "missing"
    if t < 0: return "<0"
    if t < 10: return "0-10"
    if t < 20: return "10-20"
    if t < 30: return "20-30"
    return "30+"


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(obs)),
        "mae_pct": round(float(np.mean(np.abs(obs - pred))), 4),
        "rmse_pct": round(float(np.sqrt(np.mean((obs - pred) ** 2))), 4),
        "bias_pct": round(float(np.mean(obs - pred)), 4),
    }


def _evaluate(rows: list[dict]) -> dict:
    obs = np.array([r["observed_soc_drop_pct"] for r in rows])
    phys = np.array([r["physics_soc_drop_pct"] for r in rows])
    cal = np.array([r["calibrated_soc_drop_pct"] for r in rows])
    return {
        "physics_only": _metrics(obs, phys),
        "physics_plus_ml": _metrics(obs, cal),
        "regression_count": int(np.sum(np.abs(obs - cal) > np.abs(obs - phys))),
        "regression_fraction": round(float(np.mean(np.abs(obs - cal) > np.abs(obs - phys))), 4),
    }


def evaluate(df, model) -> dict:
    rows = df.to_dict("records")
    for r in rows:
        r["calibrated_soc_drop_pct"] = predict_delta_soc_row(r, model, r["physics_soc_drop_pct"])

    result: dict = {"convention": "soc_drop_pct = soc_start - soc_end (positive); "
                                  "final = physics + ml_residual"}
    for split in ("val", "test"):
        sub = [r for r in rows if r["split"] == split]
        result[split] = {"overall": _evaluate(sub)}
        by_vehicle: dict[str, list] = {}
        by_speed: dict[str, list] = {}
        by_temp: dict[str, list] = {}
        for r in sub:
            by_vehicle.setdefault(r["vehicle_id"], []).append(r)
            by_speed.setdefault(_speed_band(r["speed_kph_mean"]), []).append(r)
            by_temp.setdefault(_temp_band(r["outside_temp_c_mean"]), []).append(r)
        result[split]["by_vehicle"] = {k: _evaluate(v) for k, v in sorted(by_vehicle.items())}
        result[split]["by_speed_band"] = {k: _evaluate(v) for k, v in sorted(by_speed.items())}
        result[split]["by_temp_band"] = {k: _evaluate(v) for k, v in sorted(by_temp.items())}

    # Promotion decision on held-out test set
    t = result["test"]["overall"]
    phys_mae = t["physics_only"]["mae_pct"]
    cal_mae = t["physics_plus_ml"]["mae_pct"]
    rel_improvement = (phys_mae - cal_mae) / phys_mae * 100.0 if phys_mae else 0.0

    # no vehicle regresses beyond +20% relative MAE
    vehicle_regressions = []
    for veh, e in result["test"]["by_vehicle"].items():
        pm = e["physics_only"]["mae_pct"]
        cm = e["physics_plus_ml"]["mae_pct"]
        if pm > 0 and (cm - pm) / pm > 0.20 and cm > pm:
            vehicle_regressions.append({"vehicle": veh, "phys_mae": pm, "cal_mae": cm, "regression_rel_pct": round((cm-pm)/pm*100, 1)})

    major_band_regressions = []
    for band, e in result["test"]["by_speed_band"].items():
        pm = e["physics_only"]["mae_pct"]
        cm = e["physics_plus_ml"]["mae_pct"]
        if pm > 0 and (cm - pm) / pm > 0.10 and cm > pm:
            major_band_regressions.append({"band": band, "phys_mae": pm, "cal_mae": cm, "regression_rel_pct": round((cm-pm)/pm*100, 1)})

    promote = (
        rel_improvement >= 5.0
        and not vehicle_regressions
        and not major_band_regressions
    )

    result["promotion"] = {
        "rel_improvement_pct": round(rel_improvement, 2),
        "phys_test_mae_pct": phys_mae,
        "cal_test_mae_pct": cal_mae,
        "vehicle_regressions": vehicle_regressions,
        "major_band_regressions": major_band_regressions,
        "decision": "PROMOTE" if promote else "KEEP_PHYSICS_ONLY",
        "rationale": ("test MAE improvement>=5% rel AND no vehicle regresses>+20% AND "
                      "no major band regresses>+10%" if promote else
                      "failed one or more promotion gates"),
    }
    return result


def write_evaluation(out_dir: Path, result: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "calibration_eval.json"
    with p.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p