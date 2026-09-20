"""Canonical vehicle context — the feature contract shared by training and serving.

This is the single source of truth for the field names a model (LoRA) and the
evaluation harness agree on. It exists to prevent *training-serving skew*: the
raw telemetry schemas differ across sources (TUM driving, EPFL charging, AURORA
live signals), so nothing below the runtime schema-adapter layer should depend on
a specific source's naming. These canonical types are the boundary every source
is normalised *into*.

Two distinct context types are modelled deliberately rather than one flat
dictionary, because the semantics differ and a naive field rename would be wrong:
``soc_arrival_pct`` (SOC when a charging session *began*) is NOT the same thing
as ``soc_pct`` (SOC at the current moment), even though both are "a SOC".

* ``VehicleDrivingContext``      — state of the vehicle at the current moment.
* ``VehicleChargingSessionContext`` — a complete charging session (historical).

A provider/harness receives one of these (never raw source telemetry) and is
responsible only for rendering it faithfully into the prompt it was trained on.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VehicleDrivingContext(BaseModel):
    """Vehicle state at the current moment (live / latest frame).

    Field names match the v2 training/eval data exactly so a grounded prompt
    composes without renaming. ``None`` means "not present in this instance" and
    is rendered as missing telemetry (the assistant must refuse rather than
    invent), never coerced to a fabricated value. Extra fields are rejected so a
    source schema error surfaces loudly instead of silently tipping fields.
    """

    model_config = ConfigDict(extra="forbid")

    soc_pct: float | None = None         # state of charge, current instant (%)
    speed_kph: float | None = None       # current speed (km/h)
    outside_temp_c: float | None = None  # ambient temperature (deg C)
    battery_temp_c: float | None = None  # HV pack temperature (deg C)
    coolant_temp_c: float | None = None  # coolant temperature (deg C)
    pack_voltage_v: float | None = None  # HV pack voltage (V)


class VehicleChargingSessionContext(BaseModel):
    """A complete (historical) charging session.

    These describe the session as a whole and must NOT be collapsed into the
    driving context: ``soc_arrival_pct`` is the SOC when the session started, not
    the vehicle's current SOC.
    """

    model_config = ConfigDict(extra="forbid")

    soc_arrival_pct: float | None = None     # SOC when the session began (%)
    soc_departure_pct: float | None = None   # SOC when the session ended (%)
    stay_min: float | None = None            # session wall-clock duration (min)
    energy_charged_kwh: float | None = None  # total energy delivered (kWh)
    max_power_kw: float | None = None        # negotiated max charge power (kW)
    max_requested_power_kw: float | None = None  # peak power request (kW)
    controlled_session: bool | None = None   # session ran under active control


class VehicleGroundTruth(BaseModel):
    """Derived / reference values used for scoring (not telemetry).

    Held separately so "what the vehicle measured" and "what we assert the
    correct answer is" never share a namespace. ``range_km`` is derived from SOC
    (``SOC/100 * nominal``), so it is treated as ground truth, not telemetry.
    """

    range_km: float | None = None  # estimated remaining range (derived, km)


__all__ = [
    "VehicleDrivingContext",
    "VehicleChargingSessionContext",
    "VehicleGroundTruth",
]
