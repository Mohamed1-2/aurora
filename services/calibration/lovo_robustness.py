"""Leave-one-vehicle-out (LOVO) grouped robustness check (WS2, final).

Model and lambda policy are FROZEN — nothing is tuned or optimised here:
* Same LightGBM hyperparameters as the frozen v2-ws1-r1 candidate.
* ``num_boost_round=40`` (the frozen early-stop best iteration). No per-fold
  early stopping, no seed changes, no hyperparameter search.
* Per-band lambda policy fixed exactly as frozen:
      0-30: 0.20 | 30-60: 1.00 | 60-90: 1.00 | 90-120: 1.00 | 120+: 0.55

Method: for each vehicle, train on ALL windows of the other 6 vehicles (whole
episodes, grouped; never both sides of a vehicle) and evaluate only on the held
out vehicle. Every prediction therefore comes from a model which never saw that
vehicle — a vehicle-safe / episode-safe independence check of the win that does
not reuse the previously-inspected test split as evaluation.

Output: pooled aggregate + per-band + per-vehicle numbers comparing physics-only
against the conservative (per-band-shrunk) predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .model import FEATURES

DEFAULT_BASE = Path("artifacts/calibration/v2-ws1")

SPEED_BANDS = ["0-30", "30-60", "60-90", "90-120", "120+"]
LAM = {"0-30": 0.20, "30-60": 1.00, "60-90": 1.00, "90-120": 1.00, "120+": 0.55}
FROZEN_ROUNDS = 40

FROZEN_PARAMS = {
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


def _band_of(speed: np.ndarray) -> np.ndarray:
    return np.select(
        [speed < 30, speed < 60, speed < 90, speed < 120],
        SPEED_BANDS[:4],
        default=SPEED_BANDS[4],
    )


def _metrics(obs, pred):
    return {
        "n": int(len(obs)),
        "mae_pct": round(float(np.mean(np.abs(obs - pred))), 4),
        "rmse_pct": round(float(np.sqrt(np.mean((obs - pred) ** 2))), 4),
        "bias_pct": round(float(np.mean(obs - pred)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    args = ap.parse_args(argv)

    base = Path(args.base)
    df = pd.read_parquet(base / "dataset_w300s.parquet")
    vehicles = sorted(df["vehicle_id"].unique())

    pooled_obs: dict[str, np.ndarray] = {b: [] for b in SPEED_BANDS + ["all"]}
    pooled_phys: dict[str, np.ndarray] = {b: [] for b in SPEED_BANDS + ["all"]}
    pooled_cons: dict[str, np.ndarray] = {b: [] for b in SPEED_BANDS + ["all"]}
    per_vehicle: dict[str, dict] = {}

    for v in vehicles:
        tr = df[df["vehicle_id"] != v]
        te = df[df["vehicle_id"] == v]

        dtr = lgb.Dataset(tr[FEATURES].astype(float), label=tr["residual_pct"].astype(float))
        model = lgb.train(FROZEN_PARAMS, dtr, num_boost_round=FROZEN_ROUNDS)

        rows = te.to_dict("records")
        corr = np.array([model.predict([[r[f] for f in FEATURES]])[0] for r in rows])
        phys = te["physics_soc_drop_pct"].to_numpy()
        obs = te["observed_soc_drop_pct"].to_numpy()
        bands = _band_of(te["speed_kph_mean"].to_numpy())
        lam = np.array([LAM[b] for b in bands])
        cons = phys + lam * corr

        per_vehicle[v] = {
            "n": int(len(te)),
            "physics_only": _metrics(obs, phys),
            "conservative": _metrics(obs, cons),
        }
        for b in SPEED_BANDS:
            m = bands == b
            if m.sum():
                pooled_obs[b].append(obs[m])
                pooled_phys[b].append(phys[m])
                pooled_cons[b].append(cons[m])
        pooled_obs["all"].append(obs)
        pooled_phys["all"].append(phys)
        pooled_cons["all"].append(cons)

    result: dict = {
        "method": "LOVO grouped (vehicle-safe); model + lambda frozen; no tuning",
        "frozen": {"num_boost_round": FROZEN_ROUNDS, "lambda_pct": LAM, "params": FROZEN_PARAMS},
        "vehicles": vehicles,
        "per_vehicle": per_vehicle,
        "pooled": {},
    }
    for b in SPEED_BANDS + ["all"]:
        o = np.concatenate(pooled_obs[b])
        p = np.concatenate(pooled_phys[b])
        c = np.concatenate(pooled_cons[b])
        result["pooled"][b] = {
            "physics_only": _metrics(o, p),
            "conservative": _metrics(o, c),
            "mae_improvement_pct": round(
                (np.mean(np.abs(o - p)) - np.mean(np.abs(o - c))) / np.mean(np.abs(o - p)) * 100.0, 2
            ),
        }

    out_path = base / "lovo_robustness.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("=== LOVO pooled (vehicle-safe, frozen model + lambda) ===")
    for b in ["all"] + SPEED_BANDS:
        row = result["pooled"][b]
        pp = row["physics_only"]["mae_pct"]
        cc = row["conservative"]["mae_pct"]
        print(f"{b:>7}: phys_MAE={pp:.4f}  cons_MAE={cc:.4f}  "
              f"improvement={row['mae_improvement_pct']:+.2f}%  n={row['physics_only']['n']}")
    print("\n=== per-vehicle ===")
    for v, r in per_vehicle.items():
        print(f"{v}: n={r['n']:>4}  phys_MAE={r['physics_only']['mae_pct']:.4f}  "
              f"cons_MAE={r['conservative']['mae_pct']:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())