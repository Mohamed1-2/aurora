"""WS1 physics energy engine (pure, deterministic, calibration-free).

This module computes energy/arrival quantities from *explicit inputs* using the
same physical model the simulator uses (controlled scenario simulator only). It
deliberately reads nothing from live vehicle state, so it can never "validate
itself" as real-world accuracy.

Scope and honesty (per AURORA WS1 decisions)
-------------------------------------------
* This is a **physics baseline** — the ML calibration (corrections against real
  held-out TUM sequences) arrives in WS2.
* ``project_arrival`` agreement with the simulator is a **software/physics
  consistency check**, NOT evidence of real-EV arrival-SOC accuracy.
* There is NO ``low_arrival_soc`` style signal here: that requires a real
  predicted ``arrival_soc_pct`` and belongs to WS4. This engine only returns a
  ``projected_arrival_soc_pct`` when the caller provides a distance.

All parameters come from the versioned ``simparams`` profile so provenance is
traceable (sourced vs assumed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .simparams import DEFAULT_PROFILE as _P, SimulatorParameterProfile


@dataclass(frozen=True)
class EnergyDrivers:
    """Breakdown of instantaneous battery power demand (W)."""
    motive_w: float      # inertia + aero + rolling at the wheel
    aero_w: float
    rolling_w: float
    inertia_w: float
    hvac_w: float
    aux_w: float
    total_w: float

    def to_dict(self) -> dict:
        return {
            "motive_w": round(self.motive_w, 2),
            "aero_w": round(self.aero_w, 2),
            "rolling_w": round(self.rolling_w, 2),
            "inertia_w": round(self.inertia_w, 2),
            "hvac_w": round(self.hvac_w, 2),
            "aux_w": round(self.aux_w, 2),
            "total_w": round(self.total_w, 2),
        }


@dataclass(frozen=True)
class ArrivalEstimate:
    distance_km: float
    projected_arrival_soc_pct: float
    energy_wh: float
    wh_per_km: float | None = None
    feasibility_note: str = ""

    def to_dict(self) -> dict:
        d = {
            "distance_km": round(self.distance_km, 2),
            "projected_arrival_soc_pct": round(self.projected_arrival_soc_pct, 2),
            "energy_wh": round(self.energy_wh, 2),
        }
        if self.wh_per_km is not None:
            d["wh_per_km"] = round(self.wh_per_km, 2)
        if self.feasibility_note:
            d["feasibility_note"] = self.feasibility_note
        return d


def _hvac_power_w(cabin_set_temp_c: float | None, ambient_temp_c: float | None) -> float:
    """HVAC simulation load. Zero when no cabin set point is provided.

    No cabin set point => No HVAC load: we never fabricate a climate demand
    (important for non-trip / replayed-telemetry paths).
    """
    if cabin_set_temp_c is None or ambient_temp_c is None:
        return 0.0
    raw = _P.value("hvac_coeff_w_per_c") * (cabin_set_temp_c - ambient_temp_c) + _P.value("hvac_base_w")
    return min(_P.value("hvac_max_w"), max(_P.value("hvac_min_w"), raw))


def _motive_power_w(speed_kph: float, accel_mps2: float = 0.0) -> tuple[float, float, float, float]:
    """Wheel drag/inertia power (W). No motor efficiency here (at the wheel)."""
    g = _P.value("g_mps2")
    mass = _P.value("mass_kg")
    crr = _P.value("crr")
    cd = _P.value("cd")
    area = _P.value("frontal_area_m2")
    rho = _P.value("rho_air_kgm3")
    v_ms = max(0.0, speed_kph) / 3.6
    p_roll = mass * g * crr * v_ms
    p_aero = 0.5 * rho * cd * area * v_ms**3
    p_inertia = mass * accel_mps2 * v_ms
    return p_roll + p_aero + p_inertia, p_aero, p_roll, p_inertia


def compute_drivers(
    speed_kph: float,
    cabin_temp_c: float | None = None,
    ambient_temp_c: float | None = None,
    accel_mps2: float = 0.0,
    profile: SimulatorParameterProfile | None = None,
) -> EnergyDrivers:
    """Instantaneous battery power demand split by driver (physics baseline)."""
    _p = profile or _P
    m_w, aero_w, roll_w, iner_w = _motive_power_w(speed_kph, accel_mps2)
    hvac_w = _hvac_power_w(cabin_temp_c, ambient_temp_c)
    aux_w = _p.value("aux_base_w")
    return EnergyDrivers(
        motive_w=m_w,
        aero_w=aero_w,
        rolling_w=roll_w,
        inertia_w=iner_w,
        hvac_w=hvac_w,
        aux_w=aux_w,
        total_w=m_w + hvac_w + aux_w,
    )


def project_arrival(
    distance_km: float,
    battery_wh: float,
    soc_pct: float,
    speed_kph: float,
    cabin_temp_c: float | None = None,
    ambient_temp_c: float | None = None,
    accel_mps2: float = 0.0,
    profile: SimulatorParameterProfile | None = None,
) -> ArrivalEstimate:
    """Project arrival SOC for a given distance using constant steady inputs.

    Consistency-of-physics warning: this uses the same model as the simulator
    (same parameter profile), so agreement is a *software consistency check*,
    not evidence of real-EV predictive accuracy. WS2 adds ML calibration on
    held-out real TUM sequences.
    """
    _p = profile or _P
    d_km = max(0.0, distance_km)
    if d_km <= 0.0:
        return ArrivalEstimate(
            distance_km=0.0,
            projected_arrival_soc_pct=max(0.0, soc_pct),
            energy_wh=0.0,
            feasibility_note="zero distance requested",
        )
    drivers = compute_drivers(
        speed_kph,
        cabin_temp_c=cabin_temp_c,
        ambient_temp_c=ambient_temp_c,
        accel_mps2=accel_mps2,
        profile=_p,
    )
    if speed_kph <= 0.5:
        return ArrivalEstimate(
            distance_km=d_km,
            projected_arrival_soc_pct=max(0.0, soc_pct),
            energy_wh=0.0,
            feasibility_note="speed ~0; cannot integrate distance",
        )
    # Time to cover the distance at the given speed.
    hours = d_km / speed_kph
    tot_wh = drivers.total_w * hours
    arrival = soc_pct - tot_wh / battery_wh * 100.0
    arrival = max(0.0, arrival)
    note = ""
    if arrival <= 0.0:
        note = "projected to reach zero battery before destination"
    return ArrivalEstimate(
        distance_km=d_km,
        projected_arrival_soc_pct=arrival,
        energy_wh=tot_wh,
        wh_per_km=tot_wh / d_km,
        feasibility_note=note,
    )


def what_if(
    distance_km: float,
    battery_wh: float,
    soc_pct: float,
    speed_kph: float,
    delta_speed_kph: float = 0.0,
    delta_cabin_temp_c: float = 0.0,
    ambient_temp_c: float | None = None,
    profile: SimulatorParameterProfile | None = None,
) -> dict:
    """Directional what-if over speed and cabin set point (physics baseline).

    Returns both the baseline projection and the perturbation so callers can
    assert monotonic direction (higher speed / hotter cabin => lower arrival
    SOC), never an "accuracy" claim.
    """
    base = project_arrival(
        distance_km=distance_km,
        battery_wh=battery_wh,
        soc_pct=soc_pct,
        speed_kph=speed_kph,
        cabin_temp_c=None,  # baseline: no HVAC (never fabricate a set point)
        ambient_temp_c=ambient_temp_c,
        profile=profile,
    )
    perturbed = project_arrival(
        distance_km=distance_km,
        battery_wh=battery_wh,
        soc_pct=soc_pct,
        speed_kph=max(0.0, speed_kph + delta_speed_kph),
        cabin_temp_c=delta_cabin_temp_c,
        ambient_temp_c=ambient_temp_c,
        profile=profile,
    )
    return {
        "baseline_arrival_soc_pct": round(base.projected_arrival_soc_pct, 2),
        "perturbed_arrival_soc_pct": round(perturbed.projected_arrival_soc_pct, 2),
        "delta_arrival_soc_pct": round(perturbed.projected_arrival_soc_pct - base.projected_arrival_soc_pct, 2),
        "delta_speed_kph": delta_speed_kph,
        "delta_cabin_temp_c": delta_cabin_temp_c,
    }