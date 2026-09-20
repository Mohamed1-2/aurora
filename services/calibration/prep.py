"""WS2-prep orchestrator: build training dataset + prep artifacts (no ML fitting).

Produces under artifacts/calibration/v2-ws1/:
* dataset_w300s.parquet      — every usable 300 s window with features + target
* split_manifest.json        — deterministic episode split (written pre-fit)
* availability_report.json   — usable windows by vehicle/split + feature missingness
* vehicle_capacity.json      — per-vehicle capacity/provenance contract
* baseline_val_test_metrics.json — physics-only MAE/RMSE (+ capacity sensitivity)

Run:
    AURORA_TELEMETRY_TUM_UDS_DIR=<raw parquet dir> python -m services.calibration.prep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .availability import build_availability, write_availability
from .baseline import compute_baseline, write_baseline
from .features import extract_features
from .physics import PhysicsBaseline
from .split import assign_split, build_manifest, summary, write_manifest
from .vehicle_capacity import capacity_for, write_capacity_table
from .windowing import WindowConfig, build_windows

ENV_UDS_DIR = "AURORA_TELEMETRY_TUM_UDS_DIR"
DEFAULT_OUT = Path("artifacts/calibration/v2-ws1")

DENSE_CORE = frozenset(
    {"soc_pct", "speed_kph", "hv_aux_power_w", "outside_temp_c", "coolant_temp_c"}
)


def resolve_uds_dir(explicit: str | None) -> Path:
    value = explicit or os.environ.get(ENV_UDS_DIR)
    if not value:
        raise SystemExit(
            f"no raw TUM parquet dir: set {ENV_UDS_DIR} or pass --uds-dir"
        )
    p = Path(value)
    if not p.is_dir():
        raise SystemExit(
            f"raw TUM parquet dir not found: {p} (refusing to fabricate; clean-503 behavior unchanged)"
        )
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WS2-prep: build calibration dataset")
    ap.add_argument("--uds-dir", default=None, help="dir containing {vehicle}.parquet")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--min-moving-fraction", type=float, default=0.8)
    args = ap.parse_args(argv)

    uds_dir = resolve_uds_dir(args.uds_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ing = _load_ingestor().TUMDrivingIngestor(uds_dir, core_signals=DENSE_CORE)

    bin_counts: dict[str, int] = {}
    all_windows = []
    for vehicle in ing.vehicles():
        bins = ing.ingest([vehicle])
        bin_counts[vehicle] = len(bins)
        all_windows.extend(bins)
        print(f"{vehicle}: 10s bins={len(bins)}", flush=True)

    cfg = WindowConfig(window_s=args.window_s, min_moving_fraction=args.min_moving_fraction)
    raw_windows = build_windows(all_windows, cfg)
    print(f"usable 300s windows={len(raw_windows)}", flush=True)

    physics = PhysicsBaseline()
    rows = []
    for w in raw_windows:
        capacity = capacity_for(w.vehicle_id)
        row = extract_features(w, physics, capacity)
        row["split"] = assign_split(w.vehicle_id)
        rows.append(row)

    manifest_rows = build_manifest(raw_windows)
    write_manifest(manifest_rows, out_dir)
    avail = build_availability(rows, bin_counts)
    write_availability(out_dir, avail)
    write_capacity_table(out_dir / "vehicle_capacity.json")

    import pandas as pd
    df = pd.DataFrame(rows)
    df_path = out_dir / "dataset_w300s.parquet"
    df.to_parquet(df_path, index=False)

    baseline = compute_baseline(rows)
    write_baseline(out_dir, baseline)

    with (out_dir / "prep_summary.json").open("w") as fh:
        json.dump({
            "window_s": args.window_s,
            "min_moving_fraction": args.min_moving_fraction,
            "uds_dir": str(uds_dir),
            "elapsed_s": round(time.time() - t0, 1),
            "n_usable_windows": len(rows),
            "split_summary": summary(manifest_rows),
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps({
        "dataset": str(df_path), "n_windows": len(df),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        "elapsed_s": round(time.time() - t0, 1),
        "baseline": str(out_dir / "baseline_val_test_metrics.json"),
    }, indent=2))
    return 0


def _load_ingestor():
    from ._bootstrap import load_tum_driving
    return load_tum_driving()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())