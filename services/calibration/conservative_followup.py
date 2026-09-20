"""Conservative-calibration follow-up (WS2, one focused investigation).

Goal: preserve the overall / 60-90 km/h gains of the LightGBM calibration
without regressing the 0-30 km/h band, then recommend promote / keep / end.

Approach (conservative by construction):
* Shrink the learned correction toward zero per speed band:
  ``final = physics + lam_band * correction`` with ``lam_band in [0,1]``
  (0 = physics-only in that band, 1 = full LightGBM correction).
* Each ``lam_band`` is tuned on the VALIDATION split only (grid 0.00..1.00,
  step 0.05, minimising band-wise val MAE). The held-out TEST split is seen a
  single time, at the end, purely for comparison — never during tuning.

This is a general, pre-planned conservative scheme ("shrink correction where it
adds little and apply it fully where it is clearly helpful"), not an
adversarial fit to the test set. Val and test behavior must agree for the
candidate to be taken seriously.

Honesty restrictions:
* No test-based early stopping, no test-side "best lambda" search, no re-tuning
  loop. The lambda policy is fully determined by validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FEATURES, train as train_lgb

DEFAULT_BASE = Path("artifacts/calibration/v2-ws1")

SPEED_BANDS = ["0-30", "30-60", "60-90", "90-120", "120+"]


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


def _overall(pred_arr, band_arr, obs):
    out = {"overall": _metrics(obs, pred_arr)}
    by_band = {}
    for b in SPEED_BANDS:
        m = band_arr == b
        if m.sum():
            by_band[b] = _metrics(obs[m], pred_arr[m])
    out["by_band"] = by_band
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    args = ap.parse_args(argv)

    base = Path(args.base)
    df = pd.read_parquet(base / "dataset_w300s.parquet")
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    res = train_lgb(df)
    model = res.model
    print(f"LightGBM best_iteration={res.best_iteration} val_residual_mae={res.val_mae_pct}%", flush=True)

    def arrays(sub):
        rows = sub.to_dict("records")
        corr = np.array([
            model.predict([[r[f] for f in FEATURES]], num_iteration=model.best_iteration)[0]
            for r in rows
        ])
        phys = sub["physics_soc_drop_pct"].to_numpy()
        obs = sub["observed_soc_drop_pct"].to_numpy()
        bands = _band_of(sub["speed_kph_mean"].to_numpy())
        return corr, phys, obs, bands

    corr_v, phys_v, obs_v, band_v = arrays(va)
    corr_t, phys_t, obs_t, band_t = arrays(te)

    # ---- tune per-band lambda on VALIDATION only ----
    grid = np.arange(0.0, 1.01, 0.05)
    lam = {}
    for b in SPEED_BANDS:
        m = band_v == b
        if m.sum() < 5:
            lam[b] = 1.0
            continue
        best = (0.0, float(np.mean(np.abs(obs_v[m] - phys_v[m]))))
        for c in grid:
            mae = float(np.mean(np.abs(obs_v[m] - (phys_v[m] + c * corr_v[m]))))
            if mae < best[1]:
                best = (c, mae)
        lam[b] = float(best[0])

    print("per-band lambda (val-tuned):", json.dumps(lam, sort_keys=True))

    # ---- apply once on test ----
    conservative = phys_t + np.array([lam[b] for b in band_t]) * corr_t

    out: dict = {
        "scheme": "per-band lambda tuned on VAL only (grid 0..1 step 0.05); test seen once for comparison",
        "per_band_lambda": lam,
        "note": "lambda=0 is physics-only; lambda=1 is full LightGBM",
    }
    out["val"] = {
        "physics_only": _overall(phys_v, band_v, obs_v),
        "full_lgb": _overall(phys_v + corr_v, band_v, obs_v),
    }
    out["test"] = {
        "physics_only": _overall(phys_t, band_t, obs_t),
        "full_lgb": _overall(phys_t + corr_t, band_t, obs_t),
        "conservative": _overall(conservative, band_t, obs_t),
    }

    def rel(a, b):
        return (a - b) / a * 100.0 if a else 0.0

    t = out["test"]
    p = t["physics_only"]["overall"]
    f = t["full_lgb"]["overall"]
    c = t["conservative"]["overall"]
    out["test"]["summary"] = {
        "phys_mae_pct": p["mae_pct"],
        "full_lgb_mae_pct": f["mae_pct"],
        "conservative_mae_pct": c["mae_pct"],
        "conservative_vs_phys_rel_pct": round(rel(p["mae_pct"], c["mae_pct"]), 2),
        "conservative_vs_full_lgb_rel_pct": round(rel(f["mae_pct"], c["mae_pct"]), 2),
    }
    band0 = t["conservative"]["by_band"]["0-30"]["mae_pct"]
    bandP0 = t["physics_only"]["by_band"]["0-30"]["mae_pct"]
    band90 = t["conservative"]["by_band"]["60-90"]["mae_pct"]
    bandP90 = t["physics_only"]["by_band"]["60-90"]["mae_pct"]
    out["test"]["goals"] = {
        "0_30_no_regression_vs_phys_pct": round(band0 - bandP0, 4),
        "0_30_conservative_mae": band0,
        "60_90_gain_preserved_pct": round(band90 - bandP90, 4),
        "60_90_conservative_mae": band90,
    }

    pth = base / "conservative_followup.json"
    with pth.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n=== TEST SUMMARY ===")
    print(json.dumps(out["test"]["summary"], indent=2))
    print("=== GOALS ===")
    print(json.dumps(out["test"]["goals"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())