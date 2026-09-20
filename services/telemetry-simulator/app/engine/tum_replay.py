"""WS1 genuine TUM replay — raw measured telemetry only.

Reads the *original raw* TUM Electric Vehicle Usage Data Set (Schreiber et al.,
TUM, CC-BY-NC-4.0) long-format parquet logs (``data/uds_data/{vehicle}.parquet``
with columns ``time``, ``value_id``, ``value``), resamples them onto the aligned
10 s window grid via ``TUMDrivingIngestor``, and emits frames that preserve:

* original wall-clock timestamps (``window_start_utc``)
* vehicle / source identity (``vehicle_id``, ``episode_id``)
* **measured fields only** — no fabricated HVAC, destination or cabin set point
* chronological ordering

Every frame is also projected through the *existing* canonical mapping
(``normalizer.from_v2_driving``) so consumers see the same context shape the
LoRA was trained on. Nothing here ever falls back to the v2 JSONL corpus — the
corpus is a training/eval artifact, not a telemetry source — and nothing is
synthesised. If the raw parquet is not available the loader raises a clean
``DataUnavailableError`` (mapped to HTTP 503), never a fabricated replay.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from vehicle_ai.normalizer import from_v2_driving

# The TUM UDS ingestor lives under services/evaluation/data/v2/ which is NOT on
# sys.path at service runtime (services/ has no __init__.py). Load it as the
# `data.v2.tum_driving` package by adding services/evaluation to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVAL_DIR = _REPO_ROOT / "services" / "evaluation"


def _load_tum_driving():
    eval_dir = str(_EVAL_DIR)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        return importlib.import_module("data.v2.tum_driving")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(f"cannot import TUM ingestor (data.v2.tum_driving): {exc}") from exc


class DataUnavailableError(RuntimeError):
    """Raw TUM parquet is not present — refuse to fabricate a replay."""


class TUMReplaySource:
    """Loads contiguous genuine-measured TUM slices from raw parquet."""

    def __init__(self, uds_dir: str | Path | None) -> None:
        self.uds_dir = Path(uds_dir) if uds_dir else None
        self._ingestor = None

    @property
    def ingestor_cls(self):
        if self._ingestor is None:
            self._ingestor = _load_tum_driving().TUMDrivingIngestor
        return self._ingestor

    # -- availability --------------------------------------------------------

    def available(self) -> bool:
        return self.uds_dir is not None and self.uds_dir.is_dir()

    def describe(self) -> dict:
        if not self.available():
            return {
                "available": False,
                "uds_dir": str(self.uds_dir) if self.uds_dir else None,
                "error": "raw TUM parquet not available; refusing to fabricate replay",
            }
        try:
            ing = self.ingestor_cls(self.uds_dir)
            vehicles = ing.vehicles()
            return {"available": True, "uds_dir": str(self.uds_dir), "vehicles": vehicles}
        except Exception as exc:  # pragma: no cover
            return {"available": False, "uds_dir": str(self.uds_dir), "error": str(exc)}

    # -- episode enumeration ---------------------------------------------------

    def episodes(self, vehicle_id: str | None = None) -> list[dict]:
        """Chronologically-ordered episode rollups (measured fields only)."""
        if not self.available():
            raise DataUnavailableError(self.describe()["error"])
        ing = self.ingestor_cls(self.uds_dir)
        windows = ing.ingest(vehicles=[vehicle_id] if vehicle_id else None)
        episodes: dict[str, list] = {}
        for w in windows:
            episodes.setdefault(w.episode_id, []).append(w)
        rollups = []
        for ep_id, wins in episodes.items():
            wins.sort(key=lambda w: w.window_start_utc)
            rollups.append(
                {
                    "episode_id": ep_id,
                    "vehicle_id": wins[0].vehicle_id,
                    "windows": len(wins),
                    "start_utc": wins[0].window_start_utc,
                    "end_utc": wins[-1].window_start_utc,
                    "signals_measured": sorted({s for w in wins for s in w.signals}),
                }
            )
        return sorted(rollups, key=lambda e: (e["vehicle_id"], e["episode_id"]))

    # -- slice selection --------------------------------------------------------

    def slice(self, vehicle_id: str, episode_id: str | None, start_index: int, duration_windows: int | None) -> list[dict]:
        """Return a contiguous measured-only replay slice (chronological)."""
        if not self.available():
            raise DataUnavailableError(self.describe()["error"])
        ing = self.ingestor_cls(self.uds_dir)
        windows = ing.ingest(vehicles=[vehicle_id])
        episode_windows: list = []
        if episode_id is not None:
            episode_windows = [w for w in windows if w.episode_id == episode_id]
            if not episode_windows:
                raise DataUnavailableError(
                    f"no episode {episode_id!r} for vehicle {vehicle_id!r}; choose from "
                    f"{sorted({w.episode_id for w in windows}) or 'no episodes'}"
                )
            episode_windows.sort(key=lambda w: w.window_start_utc)
        else:
            # Longest contiguous episode for this vehicle.
            by_ep: dict[str, list] = {}
            for w in windows:
                by_ep.setdefault(w.episode_id, []).append(w)
            if not by_ep:
                raise DataUnavailableError(f"no TUM episodes for vehicle {vehicle_id!r}")
            longest = max(by_ep.values(), key=lambda ws: len(ws))
            longest.sort(key=lambda w: w.window_start_utc)
            episode_windows = longest
            episode_id = longest[0].episode_id

        n = len(episode_windows) - start_index
        if duration_windows is not None:
            n = min(n, duration_windows)
        if n <= 0:
            raise DataUnavailableError(
                f"start_index={start_index} beyond episode {episode_id!r} ({len(episode_windows)} windows)"
            )
        selected = episode_windows[start_index : start_index + n]
        return [self._to_slice(w, i) for i, w in enumerate(selected)]

    def _to_slice(self, w, index: int) -> dict:
        signals = {k: float(v) if isinstance(v, (int, float)) else v for k, v in w.signals.items()}
        canonical = from_v2_driving(signals).model_dump(exclude_none=True)
        return {
            "vehicle_id": w.vehicle_id,
            "window_start_utc": w.window_start_utc,
            "episode_id": w.episode_id,
            "signals": signals,
            "canonical": canonical,
            "provenance": w.provenance.to_dict(),
            "index": index,
        }


def get_replay_source(uds_dir: str | Path | None) -> TUMReplaySource:
    """Entry point used by the telemetry-simulator service."""
    return TUMReplaySource(uds_dir)