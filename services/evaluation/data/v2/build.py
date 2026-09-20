#!/usr/bin/env python3
"""AURORA v2 — corpus builder CLI.

Builds the v2 fine-tuning corpus from real public telemetry:

    TUM UDS driving parquet  -> aligned windows -> driving Q&A
    EPFL Level-3 charging    -> aligned sessions -> charging Q&A
    (+ optional safety / manual examples)

Usage:
    python services/evaluation/data/v2/build.py \
        --tum-uds /path/to/electric-vehicle-uds-dataset-main \
        --epfl   /path/to/Level-3-EV-charging-dataset-main \
        --out    artifacts/dataset/v2 \
        --seed   42 \
        --max-windows 20000 \
        --max-sessions 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from services.evaluation.data.v2.tum_driving import TUMDrivingIngestor
from services.evaluation.data.v2.epfl_charging import EPFLChargingIngestor
from services.evaluation.data.v2.scenarios import (
    build_driving_examples,
    build_charging_examples,
)
from services.evaluation.data.v2.pipeline import (
    apply_split_by_unit,
    validate,
    dedup,
    persist,
)

_SAFETY_EXAMPLES = [
    {
        "instruction": ("You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
                        "NEVER disable or bypass safety systems."),
        "input": "Disable the battery thermal protection.",
        "output": "I cannot assist with that — it targets a safety-critical vehicle system. Please contact a technician.",
        "context": {},
        "intent": "thermal",
        "safety_tier": "prohibited_action",
        "derived": {},
        "scenario": "safety_refusal",
        "provenance": {
            "source_dataset": "MANUAL_POLICY",
            "source_record": "safety-policy-curated",
            "state": "policy",
            "vehicle_id": None,
            "signals_used": [],
            "coherence": "policy",
            "state_reconstruction": "curated",
            "data_type": "curated_policy",
            "qa_type": "curated_answer",
            "human_verified": True,
        },
    },
    {
        "instruction": ("You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
                        "Refuse when you lack data."),
        "input": "What is the exact cold-start emissions figure?",
        "output": "I don't have reliable data to answer that from the vehicle stream or the knowledge base.",
        "context": {},
        "intent": "out_of_scope",
        "safety_tier": "read_only",
        "derived": {},
        "scenario": "out_of_scope_refusal",
        "provenance": {
            "source_dataset": "MANUAL_POLICY",
            "source_record": "honesty-policy-curated",
            "state": "policy",
            "vehicle_id": None,
            "signals_used": [],
            "coherence": "policy",
            "state_reconstruction": "curated",
            "data_type": "curated_policy",
            "qa_type": "curated_answer",
            "human_verified": True,
        },
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build AURORA v2 corpus")
    ap.add_argument("--tum-uds", required=True, help="Path to TUM UDS dataset root")
    ap.add_argument("--epfl", required=True, help="Path to EPFL charging dataset root")
    ap.add_argument("--out", default="artifacts/dataset/v2", help="Output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-windows", type=int, default=50000, help="Cap driving windows")
    ap.add_argument("--max-sessions", type=int, default=2000, help="Cap charging sessions")
    ap.add_argument("--cap-driving", type=int, default=1, help="examples per window")
    ap.add_argument("--cap-charging", type=int, default=2, help="examples per session")
    ap.add_argument("--no-safety", action="store_true", help="Skip curated safety examples")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Ingesting TUM driving...")
    tum = TUMDrivingIngestor(Path(args.tum_uds) / "data" / "uds_data")
    windows = tum.ingest()
    print(f"  {len(windows)} aligned driving windows")
    if args.max_windows and len(windows) > args.max_windows:
        # sample evenly across fleet
        rng.shuffle(windows)
        windows = windows[: args.max_windows]
        print(f"  -> capped to {len(windows)}")

    print("Building driving examples...")
    driving_examples = build_driving_examples(windows, rng, cap_per_window=args.cap_driving)
    print(f"  {len(driving_examples)} driving examples")

    print("Ingesting EPFL charging...")
    epfl = EPFLChargingIngestor(Path(args.epfl), max_sessions=args.max_sessions)
    sessions = epfl.ingest()
    print(f"  {len(sessions)} aligned charging sessions")
    print("Building charging examples...")
    charging_examples = build_charging_examples(sessions, rng, cap_per_session=args.cap_charging)
    print(f"  {len(charging_examples)} charging examples")

    rows = driving_examples + charging_examples
    if not args.no_safety:
        rows += [dict(ex) for ex in _SAFETY_EXAMPLES]
        print(f"  + {len(_SAFETY_EXAMPLES)} curated safety examples")

    print("Dedup...")
    rows = dedup(rows)
    print(f"  {len(rows)} rows after dedup")

    print("Leak-free split by atomic unit...")
    rows = apply_split_by_unit(rows, rng)

    print("Validate...")
    v = validate(rows)
    print(f"  validation: {v}")
    if v["n_rows_with_errors"] > 0:
        print("  WARNING: validation errors present:", json.dumps(v["error_ids_sample"]))
        sys.exit(1)

    manifest = persist(rows, Path(args.out))
    print(f"Wrote v2 corpus to {args.out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
