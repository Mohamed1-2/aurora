"""Shared v2 dataset types: real-telemetry windows/sessions and provenance.

These dataclasses are deliberately lightweight (no pydantic) so the ingestors
work with stdlib + pyarrow only. Each ingestor emits normalized ``DrivingWindow``
or ``ChargingSession`` records carrying the full provenance contract required by
the AURORA v2 data-integrity rules (docs/dataset-v2-inspection-report.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Provenance:
    """Traceability for a single training example (decision #6)."""

    source_dataset: str
    source_record: str
    state: str
    vehicle_id: str | None
    signals_used: list[str] = field(default_factory=list)
    coherence: str = ""
    state_reconstruction: str = ""
    data_type: str = "real_telemetry"
    qa_type: str = "synthetic_generated"
    human_verified: bool = False

    # TUM driving
    window_start_utc: str | None = None
    window_duration_s: float | None = None
    episode_id: str | None = None

    # EPFL charging
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_dataset": self.source_dataset,
            "source_record": self.source_record,
            "state": self.state,
            "vehicle_id": self.vehicle_id,
            "signals_used": self.signals_used,
            "coherence": self.coherence,
            "state_reconstruction": self.state_reconstruction,
            "data_type": self.data_type,
            "qa_type": self.qa_type,
            "human_verified": self.human_verified,
        }
        if self.window_start_utc is not None:
            d["window_start_utc"] = self.window_start_utc
        if self.window_duration_s is not None:
            d["window_duration_s"] = self.window_duration_s
        if self.episode_id is not None:
            d["episode_id"] = self.episode_id
        if self.session_id is not None:
            d["session_id"] = self.session_id
        return d


@dataclass
class DrivingWindow:
    """One temporally-aligned multi-signal driving snapshot from the TUM parquet."""

    vehicle_id: str
    window_start_utc: str
    window_duration_s: float
    episode_id: str
    signals: dict[str, float]  # canonical names -> values (see SIGNAL_MAP)
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "window_start_utc": self.window_start_utc,
            "window_duration_s": self.window_duration_s,
            "episode_id": self.episode_id,
            "signals": self.signals,
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class ChargingSession:
    """One real EPFL charging measurement-aligned session."""

    session_id: str
    source_dataset: str = "EPFL_L3_CHARGING"
    measurements: list[dict[str, float]] = field(default_factory=list)  # aligned 1-min rows
    summary: dict[str, float] = field(default_factory=dict)
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_dataset": self.source_dataset,
            "measurements": self.measurements,
            "summary": self.summary,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
