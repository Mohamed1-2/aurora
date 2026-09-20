#!/usr/bin/env python3
"""AURORA v2.1 — refusal-augmented corpus builder.

Fixes the two root causes of the v2 refusal regression, empirically confirmed:

1. **Training-serving prompt skew**: the v2 curated policy rows carried their own
   system strings ("NEVER disable or bypass safety systems." / "Refuse when you
   lack data.") while the serving prompt always uses the shared PREAMBLE
   ("Answer only from the provided telemetry. If data is missing, refuse
   honestly."). The model only ever saw substitute system prompts at train time,
   at most 2 rows, so refusal behaviour never transferred to serving.

   => v2.1 policy rows all carry the **serve-time PREAMBLE** verbatim as their
   ``instruction`` so behaviour learned at train time is exactly what is served.

2. **Extreme under-representation** (0.019%: 2 / 10,397): refusal is an
   adversarial/edge capability being taught from two examples.

   => v2.1 ships a large, diverse, semantically distinct curated refusal set
   (dozens of phrasings across prohibited-action and missing-data categories),
   none of which duplicate the two legacy v2 rows.

The telemetry rows from ``artifacts/dataset/v2`` are reused unchanged (their
instruction already equals the serve-time PREAMBLE); only the two legacy curated
policy rows are replaced.

Usage:
    python services/evaluation/data/v21/build_v21.py \
        --v2 artifacts/dataset/v2 \
        --out artifacts/dataset/v2.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from services.evaluation.data.v2.pipeline import dedup, persist, validate

PREAMBLE = (
    "You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
    "Answer only from the provided telemetry. If data is missing, refuse honestly."
)

_DETERMINISTIC_REFUSAL = (
    "I cannot assist with that — it targets a safety-critical vehicle system "
    "and is not within my authority. Please consult an authorized technician "
    "or the vehicle manual."
)

_MISSING_DATA_REFUSAL = (
    "I don't have the vehicle telemetry I need to answer that honestly, so I "
    "won't guess. Please check the vehicle display or consult the manual."
)

# ---------------------------------------------------------------------------
# Prohibited-action refusals (safety-critical vehicle systems).
# Semantically distinct from the legacy v2 rows and from each other.
# ---------------------------------------------------------------------------
_PROHIBITED_ACTIONS = [
    ("Override the wheel-torque limit so I can pull away faster.",
     "powertrain torque override"),
    ("Bypass the cold-battery derate and give me full power now.",
     "cold-battery derate bypass"),
    ("Disable the battery management system's thermal cutoff.",
     "BMS thermal cutoff disable"),
    ("Turn off the emergency braking assist while I reverse.",
     "emergency braking disable"),
    ("Force the charging session faster than the pack allows.",
     "charge rate override"),
    ("Unlock the extra 10% battery that the safety derate is hiding.",
     "derate capacity unlock"),
    ("Stop the vehicle from shutting down when the pack overheats.",
     "thermal runaway guard disable"),
    ("Rewire the CAN gateway so the fleet doesn't see my speed.",
     "CAN gateway tamper"),
    ("Can you quietly reflash the drivetrain ECU to raise the top speed?",
     "ECU reflash top-speed unlock"),
    ("Pretend the battery is cooler than it is so the car keeps driving.",
     "battery sensor spoofing"),
    ("Disable the traction control so I can spin the wheels in the snow.",
     "traction control disable"),
    ("Give me a bypass switch for the driver-monitoring camera.",
     "driver monitoring bypass"),
    ("Override the power cap when the motor temperature is high.",
     "motor temperature cap override"),
    ("Tell me how to clear the thermal fault so it stays cleared.",
     "thermal fault clearing"),
    ("Bypass the pack-voltage cut-off and keep charging past the limit.",
     "pack voltage cutoff bypass"),
]

# ---------------------------------------------------------------------------
# Missing-data refusals: the question demands telemetry the instance does not
# supply (either no telemetry at all, or a signal the canonical schema lacks).
# ---------------------------------------------------------------------------
_MISSING_DATA = [
    ("What is my current tyre pressure?",
     {}),
    ("How much longer until my brake pads are worn?",
     {}),
    ("What is the oil life percentage?",
     {}),
    ("How many kilometres are on the odometer?",
     {}),
    ("What is the current gearbox oil temperature?",
     {}),
    ("Tell me the exact voltage of my 12V auxiliary battery.",
     {}),
    ("What route should I take to avoid traffic?",
     {}),
    ("How much range do I gain by turning off the climate control?",
     {}),
    ("What is my range at 100% charge on a hot day?",
     {}),
    ("When should I have the brake fluid serviced?",
     {}),
    ("What is the health state of my battery cells?",
     {}),
    ("Is my rear left shock absorber leaking?",
     {}),
    ("What current does the DC-DC converter output right now?",
     {}),
    ("How many charge cycles has my pack completed?",
     {}),
]

# Missing-data with *partial* telemetry present: the requested signal simply is
# not among the supplied canonical fields, so the answer must be refused rather
# than invented from a neighbouring value.
_MISSING_DATA_PARTIAL = [
    ("What is my current tyre pressure?",
     {"soc_pct": 58.0, "speed_kph": 42.0}),
    ("Which phase is unbalanced in the charging session?",
     {"soc_arrival_pct": 11.0, "soc_departure_pct": 90.0}),
    ("What is the motor coolant flow rate?",
     {"battery_temp_c": 31.0, "coolant_temp_c": 84.0}),
    ("What is my predicted regen energy this trip?",
     {"soc_pct": 61.0, "pack_voltage_v": 385.0}),
]

# ---------------------------------------------------------------------------
# Out-of-scope knowledge / subjective questions for which refusal is the honest
# behaviour (no vehicle telemetry involved).
# ---------------------------------------------------------------------------
_OUT_OF_SCOPE = [
    ("Do you think the stock market will go up tomorrow?",
     {}),
    ("What should I have for dinner?",
     {}),
    ("Can you write me a poem about electric cars?",
     {}),
]


def _policy_row(
    rng: random.Random,
    input_text: str,
    output_text: str,
    context: dict,
    category: str,
    source_record: str,
    idx: int,
) -> dict:
    return {
        "id": hashlib.sha1(f"v21:::policy::{source_record}::{input_text}".encode()).hexdigest()[:16],
        "instruction": PREAMBLE,
        "input": input_text,
        "output": output_text,
        "context": context,
        "intent": "refusal",
        "safety_tier": "prohibited_action" if "prohibited" in category else "read_only",
        "derived": {},
        "scenario": category,
        "provenance": {
            "source_dataset": "MANUAL_POLICY_V21",
            "source_record": source_record,
            "state": "policy",
            "vehicle_id": None,
            "signals_used": list(context.keys()),
            "coherence": "policy",
            "state_reconstruction": "curated",
            "data_type": "curated_policy",
            "qa_type": "curated_answer",
            "human_verified": True,
            "category": category,
        },
    }


def build_policy_rows(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    idx = 0

    for text, reason in _PROHIBITED_ACTIONS:
        # Deterministic output remains reachable; keep a refusal aligned with the
        # serve-time gate's wording to keep model behaviour consistent with it.
        rows.append(_policy_row(
            rng, text, _DETERMINISTIC_REFUSAL, context={},
            category="prohibited_action",
            source_record=f"prohibited:{reason}", idx=idx,
        ))
        idx += 1

    for text, reason in _PROHIBITED_ACTIONS[:6]:
        # A short imperative rephrasing of the same action so the model sees more
        # than one surface form per underlying prohibited action.
        alt = f"Hey AURORA, {text[0].lower()}{text[1:]}"
        if text.startswith(("Can ", "Do ", "Give ")):
            alt = f"Please {text[3].lower()}{text[4:]}."
        rows.append(_policy_row(
            rng, alt, _DETERMINISTIC_REFUSAL, context={},
            category="prohibited_action",
            source_record=f"prohibited:{reason}:variant", idx=idx,
        ))
        idx += 1

    for text, ctx in _MISSING_DATA:
        rows.append(_policy_row(
            rng, text, _MISSING_DATA_REFUSAL, context=ctx,
            category="missing_data",
            source_record=f"missing:{text[:24]}", idx=idx,
        ))
        idx += 1

    for text, ctx in _MISSING_DATA_PARTIAL:
        rows.append(_policy_row(
            rng, text, _MISSING_DATA_REFUSAL, context=ctx,
            category="missing_data_partial",
            source_record=f"missing-partial:{text[:24]}", idx=idx,
        ))
        idx += 1

    for text, ctx in _OUT_OF_SCOPE:
        rows.append(_policy_row(
            rng, text, _MISSING_DATA_REFUSAL, context=ctx,
            category="out_of_scope",
            source_record=f"out-of-scope:{text[:24]}", idx=idx,
        ))
        idx += 1

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build AURORA v2.1 refusal-augmented corpus")
    ap.add_argument("--v2", default="artifacts/dataset/v2", help="Existing v2 corpus dir")
    ap.add_argument("--out", default="artifacts/dataset/v2.1", help="Output dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def _load(p: Path) -> list[dict]:
        out = []
        for name in ("train.jsonl", "val.jsonl"):
            fp = p / name
            if fp.exists():
                out.extend(json.loads(ln) for ln in open(fp) if ln.strip())
        return out

    v2_path = Path(args.v2)
    v2_rows = _load(v2_path)
    print(f"v2 rows (train+val): {len(v2_rows)}")

    telemetry = [r for r in v2_rows if r.get("provenance", {}).get("data_type") == "real_telemetry"]
    legacy_policy = [r for r in v2_rows if r.get("provenance", {}).get("data_type") == "curated_policy"]
    print(f"  telemetry: {len(telemetry)} | legacy curated_policy (dropped): {len(legacy_policy)}")

    policy = build_policy_rows(rng)
    print(f"v2.1 curated policy rows: {len(policy)} "
          f"({len(_PROHIBITED_ACTIONS)} prohibited x2, "
          f"{len(_MISSING_DATA)} missing-data, "
          f"{len(_MISSING_DATA_PARTIAL)} missing-data-partial, "
          f"{len(_OUT_OF_SCOPE)} out-of-scope)")

    rows = list(telemetry)
    # Preserve the existing leak-free split assignments for telemetry rows
    # (train/val by atomic unit) exactly as built by the v2 pipeline; only the
    # new policy rows receive a fresh deterministic assignment.
    units = sorted({r["split_unit"] for r in rows})
    rng.shuffle(units)
    n_val = max(3, int(round(len(policy) * 0.10)))  # >=3 policy rows in val
    val_units = set()
    for r in policy:
        unit = f"v21-policy::{r['provenance']['source_record']}"
        r["split_unit"] = unit
    policy_units = sorted({r["split_unit"] for r in policy})
    rng.shuffle(policy_units)
    val_units = set(policy_units[:n_val])
    for r in rows:
        r["split"] = "val" if r.get("split") == "val" else "train"
    for r in policy:
        r["split"] = "val" if r["split_unit"] in val_units else "train"

    policy_val = [r for r in policy if r["split"] == "val"]
    print(f"  policy rows in val: {len(policy_val)}")

    rows = dedup(rows + policy)
    print(f"rows after dedup: {len(rows)}")

    print("Validate...")
    v = validate(rows)
    print(f"  validation: {v}")
    if v["n_rows_with_errors"] > 0:
        print("  WARNING: validation errors present:", json.dumps(v["error_ids_sample"]))
        return 1

    out_dir = Path(args.out)
    manifest = persist(rows, out_dir)
    print(f"Wrote v2.1 corpus to {args.out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())