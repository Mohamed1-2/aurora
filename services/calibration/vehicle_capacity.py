"""Per-vehicle usable-battery-capacity metadata (the user-approved contract).

One global capacity is never silently applied to every vehicle. Every TUM
vehicle is listed here with ``usable_capacity_kwh``, ``provenance``,
``source_reference`` and an ``assumption`` flag. The dataset README states a
fleet-level "Net Energy Capacity" of 58 kWh for both vehicle models (ID.3 Pro
Performance and CUPRA Born), but we have no per-vehicle *measured* usable
capacity, so every row is flagged ``assumption=True``.

Primary held-out metrics emphasize SOC-point error (measured). Energy/kWh
metrics are derived via these capacities and always labelled as
capacity-dependent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Capacity in kWh used when converting SOC-point deltas to Wh (positive-drop
# convention: soc_drop_pct = soc_start_pct - soc_end_pct).
DEFAULT_CAPACITY_KWH = 58.0

_PROVENANCE = "tum_dataset_readme_fleet_spec"
_SOURCE = "README.MD section 'Fleet Specifications' (Net Energy Capacity 58 kWh)"
_NOTE = "fleet-level net capacity from dataset README; no per-vehicle measured usable capacity"


@dataclass(frozen=True)
class VehicleCapacity:
    vehicle_id: str
    usable_capacity_kwh: float
    provenance: str
    source_reference: str
    assumption: bool
    notes: str = ""


def _fleet_capacity(vehicle_id: str) -> VehicleCapacity:
    return VehicleCapacity(
        vehicle_id=vehicle_id,
        usable_capacity_kwh=DEFAULT_CAPACITY_KWH,
        provenance=_PROVENANCE,
        source_reference=_SOURCE,
        assumption=True,
        notes=_NOTE,
    )


VEHICLE_CAPACITIES: dict[str, VehicleCapacity] = {
    vid: _fleet_capacity(vid)
    for vid in ("ID1", "ID2", "CUP1", "CUP2", "CUP3", "CUP4", "CUP5")
}


def capacity_for(vehicle_id: str) -> float:
    try:
        return VEHICLE_CAPACITIES[vehicle_id].usable_capacity_kwh
    except KeyError:
        raise KeyError(f"no capacity metadata for unknown TUM vehicle {vehicle_id!r}") from None


def capacity_table() -> list[dict]:
    rows = [asdict(vc) for vc in VEHICLE_CAPACITIES.values()]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]
    return [
        {"version": "v2-ws1", "capacity_kwh_default": DEFAULT_CAPACITY_KWH,
         "contract_hash": digest, "vehicles": rows}
    ]


def write_capacity_table(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(capacity_table()[0], fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path