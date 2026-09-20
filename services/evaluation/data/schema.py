"""Production data schema for the AURORA fine-tuning corpus.

This is the **canonical record shape** a production assistant would log and a
data engineer would curate.  Each row is one grounded assistant interaction,
carrying everything the fine-tune (and eval) needs:

* the raw user utterance (verbatim — this is what makes the LM realistic)
* the classified intent + safety policy tier
* the **telemetry context** — the live signals at that instant the answer grounded on
* a **human-verified grounded answer** + doc citations
* labels (verified flag, user rating) used for filtering/preference tuning
* a stable `split` (train / val / test) assigned **by vehicle**, never at random

The fixed set of intents mirrors ``assistant-api/app/core/intents.py``.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class SafetyTier(str, Enum):
    read_only = "read_only"
    safe_action = "safe_action"
    restricted_action = "restricted_action"
    prohibited_action = "prohibited_action"


#: Fixed intent vocabulary — kept in sync with app/core/intents.py.
INTENTS = (
    "range",
    "battery",
    "charging",
    "tire",
    "diagnostics",
    "thermal",
    "energy",
    "maintenance",
    "speed",
    "driving_mode",
    "winter",
    "summary",
    "greeting",
    "out_of_scope",
)


class AnchoredSignal(BaseModel):
    """A single live signal the answer grounded on."""

    name: str
    value: float
    unit: str


class TelemetryContext(BaseModel):
    """The live-vehicle snapshot for a single interaction."""

    vehicle_id: str
    timestamp: float
    soc_pct: float
    range_km: float
    speed_kph: float
    battery_temp_c: float
    coolant_temp_c: float
    motor_power_kw: float
    regen_power_kw: float
    fl_kpa: float
    fr_kpa: float
    rl_kpa: float
    rr_kpa: float
    outside_temp_c: float
    dtcs: list[str] = Field(default_factory=list)
    charging: bool = False

    def anchored(self, names: list[str]) -> list[AnchoredSignal]:
        """Return only the signals that map to a known anchor attribute."""
        mapping = {
            "distance_to_empty_km": ("range_km", "km"),
            "battery_soc_pct": ("soc_pct", "%"),
            "battery_temp_c": ("battery_temp_c", "degC"),
            "coolant_temp_c": ("coolant_temp_c", "degC"),
            "motor_power_kw": ("motor_power_kw", "kW"),
            "regen_power_kw": ("regen_power_kw", "kW"),
            "tire_pressure_fl_kpa": ("fl_kpa", "kPa"),
            "tire_pressure_fr_kpa": ("fr_kpa", "kPa"),
            "tire_pressure_rl_kpa": ("rl_kpa", "kPa"),
            "tire_pressure_rr_kpa": ("rr_kpa", "kPa"),
            "outside_temp_c": ("outside_temp_c", "degC"),
            "vehicle_speed_kph": ("speed_kph", "km/h"),
        }
        out: list[AnchoredSignal] = []
        for n in names:
            if n in mapping:
                attr, unit = mapping[n]
                out.append(AnchoredSignal(name=n, value=getattr(self, attr), unit=unit))
        return out


class DatasetRow(BaseModel):
    """One curated interaction in the fine-tuning corpus."""

    id: str
    utterance: str
    intent: str
    safety_tier: SafetyTier
    refusals: bool = False
    context: TelemetryContext
    anchor_signals: list[str] = Field(default_factory=list)
    grounded_answer: str
    citations: list[str] = Field(default_factory=list)
    verified: bool = True
    user_rating: int | None = Field(default=None, ge=0, le=5)
    source: str = "simulated-production"
    split: Split


def row_id(utterance: str, context: Any) -> str:
    """Deterministic id so curation is reproducible."""
    raw = f"{utterance}::{context.vehicle_id}::{context.timestamp:.0f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def to_jsonl(rows: list[DatasetRow]) -> str:
    return "\n".join(r.model_dump_json() for r in rows) + "\n"
