"""Fit the calibration model and evaluate held-out performance (WS2).

Reads the prep dataset, trains LightGBM (early-stop on val, untouched test),
evaluates physics-only vs physics+calibration on val/test, saves the model +
meta + evaluation. Training is deterministic (seeded).

Run:
    .venv/bin/python -m services.calibration.fit
    artifact base dir defaults to artifacts/calibration/v2-ws1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .evaluate import evaluate, write_evaluation
from .model import save_model, train

DEFAULT_BASE = Path("artifacts/calibration/v2-ws1")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    args = ap.parse_args(argv)

    base = Path(args.base)
    t0 = time.time()
    df = pd.read_parquet(base / "dataset_w300s.parquet")

    print(f"train rows: {len(df[df['split']=='train'])} "
          f"| val: {len(df[df['split']=='val'])} | test: {len(df[df['split']=='test'])}", flush=True)

    res = train(df)
    print(f"best_iteration={res.best_iteration}  val MAE (residual)={res.val_mae_pct}%")

    model = res.model
    save_model(res, base)
    eval_result = evaluate(df, model)
    write_evaluation(base, eval_result)

    prom = eval_result["promotion"]
    print(json.dumps({
        "best_iteration": res.best_iteration,
        "val_residual_mae_pct": res.val_mae_pct,
        "test_phys_mae_pct": prom["phys_test_mae_pct"],
        "test_cal_mae_pct": prom["cal_test_mae_pct"],
        "rel_improvement_pct": prom["rel_improvement_pct"],
        "decision": prom["decision"],
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())