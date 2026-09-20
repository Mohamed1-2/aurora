"""Real-world telemetry source for the AURORA fine-tuning corpus.

The offline corpus generator (``generate.py``) normally stands in for mined
production telemetry with a seeded random ``TelemetryContext``.  This module
replaces that mock with **real field data** from the *Electric Vehicle Usage
Data Set for Battery Aging Studies* (Schreiber et al., TUM Institute of
Automotive Technology, CC-BY-NC-4.0) — 72 000+ km of UDS/OBD-II logs from seven
series-production BEVs (VW ID.3 / CUPRA Born).

We consume the published **JSON histograms** (``data/json/track_{vehicle}_{value_id}.json``)
so the ingestor needs **no pandas / pyarrow** — just the stdlib — and works fully
offline.  Each file bins one signal across real driving; we turn ``x_values`` /
``y_values`` into an **empirical sampler** and draw values from it.  Missing
signals (tire pressure, DTCs, drive/regen power) have no counterpart in the
field data, so we fall back to a small physics/profile prior — that is a
documented limitation, not a silent assumption.

Reference:
    Schreiber, M., Köning, L., Balke, G., Abo Gamra, K., et al.
    "Lab-to-Field Gap in Battery Aging Studies". *eTransportation*, 27, 100518.
    https://github.com/TUMFTM/electric-vehicle-uds-dataset
"""

from __future__ import annotations

import json
import random
from pathlib import Path

try:
    from .schema import TelemetryContext  # when run as a package module
except ImportError:  # when imported directly
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import TelemetryContext  # noqa: E402

# value_id -> TelemetryContext field we ground from real UDS field data.
# Chosen because every vehicle in the set exposes them as driving histograms.
_VALUE_ID_TO_FIELD = {
    900: "soc_pct",        # hv_soc [%]
    4: "speed_kph",        # vehicle_speed [km/h]
    15: "outside_temp_c",  # ambient_air_temp [°C]
    1269: "coolant_temp_c",  # coolant_temp_inverter_inlet [°C]
    1208: "battery_temp_c",  # hv_temp_min [°C]  (coldest cell)
}

# Signals the field set has **no** counterpart for -> physics/profile prior.
_PROFILE_FIELDS = ("fl_kpa", "fr_kpa", "rl_kpa", "rr_kpa", "motor_power_kw",
                   "regen_power_kw", "range_km")

_RANGE_PER_100_SOC = 400.0  # km at full charge (ORION-X nominal ~58 kWh)


class _EmpiricalSampler:
    """Weighted sampler over a real histogram (bin centers weighted by counts)."""

    __slots__ = ("centers", "weights", "_total")

    def __init__(self, x_values: list[float], y_values: list[float]) -> None:
        centers: list[float] = []
        weights: list[float] = []
        for lo, hi, w in zip(x_values, x_values[1:], y_values):
            if w > 0:
                centers.append((lo + hi) / 2.0)
                weights.append(float(w))
        self.centers = centers
        self.weights = weights
        self._total = sum(weights)

    def sample(self, rng: random.Random) -> float:
        if self._total <= 0 or not self.centers:
            raise ValueError("empty histogram")
        r = rng.uniform(0, self._total)
        acc = 0.0
        last = self.centers[0]
        for c, w in zip(self.centers, self.weights):
            acc += w
            last = c
            if r <= acc:
                return round(c + rng.uniform(-0.5, 0.5), 1)
        return round(last, 1)


class TelemetrySource:
    """Empirical, real-world distributions indexed by signal name."""

    def __init__(self, field_map: dict[str, _EmpiricalSampler]) -> None:
        self.field_map = field_map

    @property
    def available(self) -> list[str]:
        return sorted(self.field_map)


def value_overview(telemetry_dir: Path | str) -> dict[int, str]:
    """Parse ``value_overview.csv`` -> {value_id: variable_name}."""
    import csv

    out: dict[int, str] = {}
    root = Path(telemetry_dir)
    # Accept either the dataset root or its `data/` subfolder.
    p = root / "data" / "value_overview.csv"
    if not p.exists():
        p = root / "value_overview.csv"
    if not p.exists():
        return out
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["value_id"])] = row["variable_name"]
            except (KeyError, ValueError):
                continue
    return out


def load_telemetry_source(telemetry_dir: Path | str) -> TelemetrySource:
    """Aggregate per-signal histograms across the 7 field vehicles."""

    telemetry_dir = Path(telemetry_dir)
    # Accept either the dataset root (`data/json`) or a direct `json` folder.
    json_dir = telemetry_dir / "data" / "json"
    if not json_dir.is_dir():
        json_dir = telemetry_dir / "json"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"no telemetry json folder under {telemetry_dir}")
    hist: dict[str, list[tuple[list[float], list[float]]]] = {
        field: [] for field in _VALUE_ID_TO_FIELD.values()
    }

    for path in json_dir.glob("track_*.json"):
        vid = _value_id_from_name(path.name)
        field = _VALUE_ID_TO_FIELD.get(vid)
        if field is None:
            continue
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        hist[field].append((doc["data"]["x_values"], doc["data"]["y_values"]))

    samplers: dict[str, _EmpiricalSampler] = {}
    for field, blocks in hist.items():
        all_x: list[float] = []
        all_y: list[float] = []
        for x, y in blocks:
            all_x.extend(x)
            all_y.extend(y)
        # Merge by re-binning is avoided: concatenating bin-center/count pairs and
        # sampling a weighted centroid across files is a faithful approximation.
        samplers[field] = _EmpiricalSampler(all_x, all_y)

    missing = [f for f, s in samplers.items() if not s.centers]
    # A core signal absent everywhere would silently produce garbage rows.
    # surface any unexpectedly empty sampler so the caller can decide.
    for f in missing:
        raise ValueError(f"telemetry source has no data for signal '{f}'")
    return TelemetrySource(samplers)


def _value_id_from_name(name: str) -> int | None:
    # track_CUP1_1208.json -> 1208
    stem = name.rsplit("_", 1)[-1]
    if not stem.endswith(".json") or not stem[:-5].isdigit():
        return None
    return int(stem[:-5])


def _profile_context(rng: random.Random, vehicle_id: str, t: float) -> TelemetryContext:
    """Return a context with only physics/profile prior fields set; real fields
    are filled in by :func:`sample_context`."""

    return TelemetryContext(
        vehicle_id=vehicle_id,
        timestamp=t,
        soc_pct=0.0,
        range_km=round(rng.uniform(40, 420), 1),
        speed_kph=0.0,
        battery_temp_c=0.0,
        coolant_temp_c=0.0,
        motor_power_kw=round(rng.uniform(-5, 60), 1),
        regen_power_kw=round(-rng.uniform(0, 30), 1),
        fl_kpa=round(rng.uniform(210, 260), 1),
        fr_kpa=round(rng.uniform(210, 260), 1),
        rl_kpa=round(rng.uniform(210, 260), 1),
        rr_kpa=round(rng.uniform(210, 260), 1),
        outside_temp_c=0.0,
        dtcs=([] if rng.random() < 0.7 else [rng.choice(["P0B36", "P0420", "C0031"])]),
        charging=rng.random() < 0.3,
    )


def sample_context(rng: random.Random, source: TelemetrySource,
                   vehicle_id: str, t: float) -> TelemetryContext:
    """Build a ``TelemetryContext`` whose core signals are drawn from the real
    field distributions, with profile-only signals from the physics prior."""

    ctx = _profile_context(rng, vehicle_id, t)
    fm = source.field_map

    for value_id, field in _VALUE_ID_TO_FIELD.items():
        sampler = fm.get(field)
        if sampler is not None:
            setattr(ctx, field, sampler.sample(rng))

    # range_km has no raw field; derive it from the *real* SOC so the answer and
    # the context stay mutually consistent and grounded.
    ctx.range_km = round(ctx.soc_pct / 100.0 * _RANGE_PER_100_SOC, 1)
    return ctx
