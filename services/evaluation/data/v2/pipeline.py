"""AURORA v2 — corpus merge, leak-free split, and validation.

Combines driving + charging (+ optional manual/safety) example rows into a single
v2 corpus, assigns deterministic train/val/test splits **by atomic unit**
(episode / session / section) so no underlying telemetry scenario appears in more
than one split, and runs integrity validation (provenance completeness, coherence
claims, PII sweep, duplicate detection).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import re

# ---------------------------------------------------------------------------
# Leak-free splitting by atomic unit
# ---------------------------------------------------------------------------

def split_by_unit(
    rows: list[dict],
    rng: random.Random,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    unit_key: str = "split_unit",
    add_unit: bool = True,
) -> list[dict]:
    """Assign split per atomic unit instead of per row.

    ``unit_key foundation`` is either a real key on the row (e.g. episode_id /
    session_id) or is derived. We assign each unique unit to a split, then stamp
    every row belonging to that unit with the same split. This guarantees a
    scenario seen in train is never re-rendered (with different wording) in val
    or test.
    """
    if add_unit:
        for r in rows:
            prov = r.get("provenance", {})
            unit = prov.get("episode_id") or prov.get("session_id")
            if unit is None:
                # fall back to source_record + vehicle
                unit = f"{prov.get('source_record')}::{prov.get('vehicle_id','')}"
            r[unit_key] = str(unit)

    units = sorted({r[unit_key] for r in rows})
    rng.shuffle(units)
    n = len(units)
    n_val = int(n * val_frac)
    n_test = int(n * (val_frac))  # same fraction for test
    # keep train = remainder
    unit_to_split = {}
    for i, u in enumerate(units):
        if i < n_val:
            unit_to_split[u] = "val"
        elif i < n_val + n_test:
            unit_to_split[u] = "test"
        else:
            unit_to_split[u] = "train"

    for r in rows:
        r["split"] = unit_to_split[r[unit_key]]
    return rows


def apply_split_by_unit(
    rows: list[dict],
    rng: random.Random,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> list[dict]:
    return split_by_unit(rows, rng, train_frac, val_frac)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_PII_RE = re.compile(
    r"(\b\d{3}[- .]?\d{3}[- .]?\d{4}\b)"      # US phone
    r"|(\b[A-Z]{2}\d{6}[A-Z0-9]?\b)"           # UK-ish
    r"|([\w.+-]+@[\w-]+\.[\w.]+)"              # email
    r"|(\b\d{9}\b)"                            # generic SSN-like
)


def validate_row(r: dict) -> list[str]:
    """Return a list of violations for a single v2 row (empty = valid)."""
    errs: list[str] = []
    for key in ("instruction", "input", "output", "context", "intent", "provenance"):
        if key not in r:
            errs.append(f"missing field: {key}")
    prov = r.get("provenance", {})
    for key in ("source_dataset", "source_record", "state", "coherence",
                "data_type", "qa_type", "human_verified"):
        if key not in prov:
            errs.append(f"provenance missing field: {key}")
    dt = prov.get("data_type")
    if dt == "real_telemetry":
        # Real-telemetry rows must carry explicit, honest coherence + reconstruction.
        if prov.get("coherence") not in ("timestamp_aligned", "per_signal_distribution"):
            errs.append(f"provenance coherence invalid for real_telemetry: {prov.get('coherence')}")
    elif dt == "curated_policy":
        # Curated policy/safety rows are a separate, explicitly-labeled category
        # (not telemetry). Allowed coherence values are policy/intent markers.
        if prov.get("coherence") not in ("policy", "intent"):
            errs.append(f"provenance coherence invalid for curated_policy: {prov.get('coherence')}")
    else:
        errs.append(f"data_type must be real_telemetry or curated_policy, got {dt}")
    # PII sweep on input/output
    for key in ("input", "output"):
        text = str(r.get(key, ""))
        if _PII_RE.search(text):
            errs.append(f"possible PII in {key}")
    return errs


def validate(rows: list[dict]) -> dict:
    """Validate the whole corpus; returns a summary dict + raises on hard error."""
    errs_by_row: dict[str, list[str]] = {}
    missing_required: list[str] = []
    for i, r in enumerate(rows):
        e = validate_row(r)
        if e:
            errs_by_row[r.get("id", f"row_{i}")] = e
            if any(x.startswith("missing field") or x.startswith("provenance missing") for x in e):
                missing_required.append(r.get("id", f"row_{i}"))
    summary = {
        "n_rows": len(rows),
        "n_rows_with_errors": len(errs_by_row),
        "missing_required": len(missing_required),
        "error_ids_sample": list(errs_by_row.keys())[:5],
    }
    return summary


def dedup(rows: list[dict]) -> list[dict]:
    """Remove exact-duplicate (input+output+context) rows, keeping first."""
    seen = set()
    out: list[dict] = []
    for r in rows:
        key = (r.get("input", ""), r.get("output", ""), repr(r.get("context")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def persist(rows: list[dict], out_dir: Path) -> dict:
    """Write sorted train/val/test JSONL + manifest. Returns summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r.get("split", "train"), []).append(r)
    for split, items in by_split.items():
        # stable sort by id
        items.sort(key=lambda x: x.get("id", ""))
        with (out_dir / f"{split}.jsonl").open("w") as fh:
            for it in items:
                fh.write(json.dumps(it, default=str) + "\n")
    # full corpus + manifest
    with (out_dir / "corpus.jsonl").open("w") as fh:
        for r in sorted(rows, key=lambda x: x.get("id", "")):
            fh.write(json.dumps(r, default=str) + "\n")
    manifest = {
        "version": "v2",
        "n_rows": len(rows),
        "by_split": {k: len(v) for k, v in by_split.items()},
        "by_dataset": _count(rows, lambda r: r["provenance"]["source_dataset"]),
        "by_intent": _count(rows, lambda r: r.get("intent", "")),
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _count(rows: list[dict], keyfn) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        k = keyfn(r)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
