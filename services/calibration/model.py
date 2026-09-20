"""Calibration-model training + inference contract (WS2).

Model: LightGBM gradient boosting regressor.
Target: the *residual* between measured and physics SOC drop
        (``residual_pct = observed_soc_drop_pct - physics_soc_drop_pct``).
Final prediction is additive and never replaces physics:
        ``final_soc_drop_pct = physics_soc_drop_pct + model.predict(features)``

Training is on whole vehicles only (train split); early stopping uses the
whole-vehicle validation split (ID1); the held-out test split (ID2, CUP3) is
evaluated once and never touched while fitting. Mirrors the approved design doc
and the promotion rule in ``baseline``/``evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

# Feature set. Physics-residual gets its own slot so the model can learn a
# magnitude-scaling correction, while telemetry-derived features add context.
FEATURES = [
    "speed_kph_mean", "speed_kph_std", "speed_kph_max", "speed_kph_p90",
    "distance_km", "accel_mean_mps2", "decel_fraction", "regen_decel_energy_kwh",
    "hv_aux_power_w_mean", "outside_temp_c_mean", "coolant_temp_c_mean",
    "battery_temp_inlet_c_mean", "battery_temp_outlet_c_mean",
    "motor_stator_temp_c_mean", "motor_rotor_temp_c_mean", "pack_voltage_v_mean",
    "physics_soc_drop_pct",
]


@dataclass(frozen=True)
class TrainResult:
    model: Any
    params: dict
    features: list[str]
    split: dict[str, list[str]]
    val_mae_pct: float
    best_iteration: int
    feature_importance: dict[str, float]


def _train_matrix(df, score_only: bool = False):
    X = df[FEATURES].astype(float)
    y = df["residual_pct"].astype(float)
    return X, y


def _default_params() -> dict:
    return {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
        "num_threads": 0,
    }


def train(df, out_dir: Path | None = None, params: dict | None = None) -> TrainResult:
    """Fit on train, early-stop on val, return a ready-to-serialize result."""
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    X_tr, y_tr = _train_matrix(train_df)
    X_val, y_val = _train_matrix(val_df)

    p = {**_default_params(), **(params or {})}
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_val, label=y_val, reference=dtr)

    model = lgb.train(
        p,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration or model.best_iteration)
    val_mae = float(np.mean(np.abs(y_val.to_numpy() - val_pred)))

    importance = {
        k: float(v) for k, v in zip(model.feature_name(), model.feature_importance("gain"))
    }
    res = TrainResult(
        model=model,
        params={k: v for k, v in p.items()},
        features=list(FEATURES),
        split={"train": ["CUP1", "CUP2", "CUP4", "CUP5"], "val": ["ID1"], "test": ["ID2", "CUP3"]},
        val_mae_pct=round(val_mae, 4),
        best_iteration=int(model.best_iteration or model.best_iteration),
        feature_importance=dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    )
    return res


def predict_delta_soc_row(features: dict, model, physics_soc_drop_pct: float) -> float:
    """Additive inference contract.

    Returns ``final_soc_drop_pct`` = physics + calibrated residual. Input must
    contain all FEATURES keys (physics_soc_drop_pct should equal the caller's
    physics value).
    """
    row = {k: float(features.get(k, 0.0)) for k in FEATURES}
    row["physics_soc_drop_pct"] = float(physics_soc_drop_pct)
    X = np.array([[row[k] for k in FEATURES]], dtype=float)
    correction = float(model.predict(X, num_iteration=model.best_iteration)[0])
    return float(physics_soc_drop_pct) + correction


def save_model(result: TrainResult, out_dir: Path) -> dict:
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "calibration_model.lgb"
    meta = {
        "version": "v2-ws1-r1",
        "model_path": str(model_path),
        "features": result.features,
        "split": result.split,
        "val_mae_pct": result.val_mae_pct,
        "best_iteration": result.best_iteration,
        "params": result.params,
        "feature_importance": result.feature_importance,
        "target": "soc_drop_pct; residual = observed - physics",
        "inference": "final_soc_drop_pct = physics_soc_drop_pct + model(features)",
        "promotion_rule": "test MAE improvement >=5% rel, no vehicle regresses >+20%, "
                          "no material harm in major bands, else keep physics-only",
        "capacity_kwh": 58.0,
    }
    # LightGBM Booster has its own save_model for portability
    result.model.save_model(str(model_path))
    with (out_dir / "model_meta.json").open("w") as fh:
        import json
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return meta


def load_model(model_dir: Path):
    """Load the saved LightGBM Booster + metadata."""
    import json
    booster = lgb.Booster(model_file=str(model_dir / "calibration_model.lgb"))
    with (model_dir / "model_meta.json").open() as fh:
        meta = json.load(fh)
    return booster, meta