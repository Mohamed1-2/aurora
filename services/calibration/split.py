"""Deterministic, leak-free episode split (user-approved rules).

Split assignment is a pure function of ``vehicle_id`` — no random sampling,
therefore deterministic by construction and impossible to leak: no window or
episode of a held-out vehicle ever reaches training. The manifest records every
episode (run) with its split BEFORE any model fitting.

Train: CUP1, CUP2, CUP4, CUP5   Val: ID1   Test: ID2, CUP3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

SPLIT_RULES: dict[str, list[str]] = {
    "train": ["CUP1", "CUP2", "CUP4", "CUP5"],
    "val": ["ID1"],
    "test": ["ID2", "CUP3"],
}

_VALID_SPLITS = frozenset(SPLIT_RULES)
_SPLIT_BY_VEHICLE = {v: s for s, vehicles in SPLIT_RULES.items() for v in vehicles}


def assign_split(vehicle_id: str) -> str:
    try:
        return _SPLIT_BY_VEHICLE[vehicle_id]
    except KeyError:
        raise KeyError(f"no split rule for unknown TUM vehicle {vehicle_id!r}") from None


@dataclass
class EpisodeRow:
    vehicle_id: str
    run_id: str
    split: str
    n_windows: int
    n_bins_used: int
    start_utc: str
    end_utc: str
    source: str = "TUM_UDS"


def build_manifest(windows: Iterable) -> list[EpisodeRow]:
    """Aggregate RawWindows into one manifest row per episode (run)."""
    runs: dict[str, dict] = {}
    for w in windows:
        r = runs.setdefault(w.run_id, {
            "vehicle_id": w.vehicle_id, "run_id": w.run_id,
            "n_windows": 0, "n_bins_used": 0, "start_utc": w.start_utc, "end_utc": w.end_utc,
        })
        r["n_windows"] += 1
        r["n_bins_used"] += w.n_bins
        r["start_utc"] = min(r["start_utc"], w.start_utc)
        r["end_utc"] = max(r["end_utc"], w.end_utc)
    rows = []
    for v in runs.values():
        v["split"] = assign_split(v["vehicle_id"])
        rows.append(EpisodeRow(**v))
    return rows


def summary(rows: list[EpisodeRow]) -> dict:
    by_split: dict[str, int] = {}
    by_vehicle: dict[str, dict] = {}
    for r in rows:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        d = by_vehicle.setdefault(r.vehicle_id, {"vehicle_id": r.vehicle_id, "split": r.split,
                                                 "n_windows": 0, "n_runs": 0})
        d["n_windows"] += r.n_windows
        d["n_runs"] += 1
    return {
        "rules": SPLIT_RULES,
        "episode_count_by_split": by_split,
        "by_vehicle": [by_vehicle[v] for v in sorted(by_vehicle)],
        "note": "split is a pure function of vehicle_id (deterministic); episodes are never split across splits",
    }


def write_manifest(rows: list[EpisodeRow], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v2-ws1-r1",
        "split_rules": SPLIT_RULES,
        "summary": summary(rows),
        "episodes": [asdict(r) for r in rows],
    }
    p = out_dir / "split_manifest.json"
    with p.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p