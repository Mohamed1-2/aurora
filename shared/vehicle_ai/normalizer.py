"""Runtime schema adapter — normalise heterogeneous telemetry into the canonical
vehicle context that models are trained and served on.

Purpose (training-serving skew prevention)
------------------------------------------
The LoRA was trained on the v2 TUM/EPFL *canonical* context names
(``soc_pct``, ``speed_kph``, ``soc_arrival_pct``, ``stay_min``, ...). Raw runtime
telemetry has different names per source: the AURORA simulator emits ORION-X
fields (``battery_soc_pct``, ``vehicle_speed_kph``, ``distance_to_empty_km``, ...),
while v2 rows already carry canonical names. If a model is served with the wrong
shape it silently answers from the wrong schema — the exact skew we must avoid.

This module is the *only* place that translates a source schema *into* the
canonical types (``VehicleDrivingContext`` / ``VehicleChargingSessionContext``).
A field is only mapped when it means the same thing; ambiguous fields (e.g.
EPFL's ``soc_arrival_pct`` vs a live ``soc_pct``) are never conflated.
"""

from __future__ import annotations

from typing import Callable, TypeVar, Union

from .canonical import (
    VehicleChargingSessionContext,
    VehicleDrivingContext,
    VehicleGroundTruth,
)

# A raw telemetry payload is a flat dict of scalar signal values, possibly with a
# "signals" container (AURORA live frames) or "summary"/"context" container (v2).
# We accept the union of shapes and map defensively.
RawTelemetry = dict

CanonicalContext = Union[VehicleDrivingContext, VehicleChargingSessionContext]

Normalizer = Callable[[RawTelemetry], CanonicalContext]

# ---------------------------------------------------------------------------
# v2 -> canonical (the source the model was trained on). These names already
# ARE canonical, so this is an identity/attribute-copy pass. It exists so the
# serving path has a single entry point regardless of source.
# ---------------------------------------------------------------------------


def from_v2_driving(raw: RawTelemetry) -> VehicleDrivingContext:
    return VehicleDrivingContext(
        soc_pct=_num(raw, "soc_pct"),
        speed_kph=_num(raw, "speed_kph"),
        outside_temp_c=_num(raw, "outside_temp_c"),
        battery_temp_c=_num(raw, "battery_temp_c"),
        coolant_temp_c=_num(raw, "coolant_temp_c"),
        pack_voltage_v=_num(raw, "pack_voltage_v"),
    )


def from_v2_charging(raw: RawTelemetry) -> VehicleChargingSessionContext:
    ctrl = raw.get("controlled_session")
    return VehicleChargingSessionContext(
        soc_arrival_pct=_num(raw, "soc_arrival_pct"),
        soc_departure_pct=_num(raw, "soc_departure_pct"),
        stay_min=_num(raw, "stay_min"),
        energy_charged_kwh=_num(raw, "energy_charged_kwh"),
        max_power_kw=_num(raw, "max_power_kw"),
        max_requested_power_kw=_num(raw, "max_requested_power_kw"),
        controlled_session=bool(ctrl) if ctrl is not None else None,
    )


def ground_truth_from_v2(raw: RawTelemetry) -> VehicleGroundTruth:
    derived = raw.get("derived", {}) if isinstance(raw.get("derived"), dict) else {}
    rv = derived.get("range_km")
    if isinstance(rv, dict):
        rv = rv.get("value")
    return VehicleGroundTruth(range_km=_num({"range_km": rv}, "range_km"))


# ---------------------------------------------------------------------------
# ORION-X live runtime -> canonical driving context.
#
# The AURORA simulator / assistant grounds answers with these legacy names:
#   battery_soc_pct, vehicle_speed_kph, outside_temp_c, battery_temp_c,
#   coolant_temp_c, battery_voltage_v.
# Each maps to the semantically-identical canonical name. There is no range in
# the canonical driving context yet; ORION-X's distance_to_empty_km is a live
# measurement, distinct from v2's derived range_km, so it is NOT force-mapped.
# ---------------------------------------------------------------------------


def from_orionx_live(raw: RawTelemetry) -> VehicleDrivingContext:
    signals = raw.get("signals", {}) if isinstance(raw.get("signals"), dict) else raw
    return VehicleDrivingContext(
        soc_pct=_num(signals, "battery_soc_pct"),
        speed_kph=_num(signals, "vehicle_speed_kph"),
        outside_temp_c=_num(signals, "outside_temp_c"),
        battery_temp_c=_num(signals, "battery_temp_c"),
        coolant_temp_c=_num(signals, "coolant_temp_c"),
        pack_voltage_v=_num(signals, "battery_voltage_v"),
    )


# ---------------------------------------------------------------------------
# Generic dispatch: pick the right normalizer from a source tag.
# ---------------------------------------------------------------------------

_DRIVING_NORMALIZERS: dict[str, Normalizer] = {
    "TUM_UDS": from_v2_driving,
    "v2_driving": from_v2_driving,
    "orionx_live": from_orionx_live,
    "orion_x": from_orionx_live,
}

_CHARGING_NORMALIZERS: dict[str, Normalizer] = {
    "EPFL_L3_CHARGING": from_v2_charging,
    "v2_charging": from_v2_charging,
}


def normalizer_for(
    source: str, *, charging: bool = False
) -> Normalizer:
    """Return the canonicalising normalizer for a telemetry source.

    Raises ``KeyError`` for an unknown source rather than silently guessing —
    a wrong schema must surface loudly, not quietly skew a model's inputs.
    """
    table = _CHARGING_NORMALIZERS if charging else _DRIVING_NORMALIZERS
    try:
        return table[source]
    except KeyError:
        raise KeyError(
            f"no normalizer registered for source={source!r} "
            f"(charging={charging}); register one in vehicle_ai.normalizer"
        ) from None


def _num(raw: RawTelemetry, key: str) -> float | None:
    v = raw.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


__all__ = [
    "CanonicalContext",
    "Normalizer",
    "VehicleDrivingContext",
    "VehicleChargingSessionContext",
    "VehicleGroundTruth",
    "from_v2_driving",
    "from_v2_charging",
    "from_orionx_live",
    "normalizer_for",
]
