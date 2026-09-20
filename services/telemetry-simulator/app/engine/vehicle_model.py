"""Physics-lite vehicle model for the ORION-X test platform.

This is not a real vehicle, but it *is* a real physical simulation:
traction power is computed from inertia + rolling resistance + aero drag,
battery state is drained by the powertrain demand, regen recovers energy,
temperatures respond to load, and the range estimator learns from rolling
Wh/km — exactly the signals an in-vehicle AI assistant would reason over.

WS1 energy accounting (single ledger)
--------------------------------------
Every per-tick battery drain is summed exactly once into
``battery_wh_tick``:

    battery_wh_tick == motive_wh_tick + hvac_wh_tick + aux_wh_tick

* ``motive_wh_tick`` — powertrain (inertia + aero + rolling plus motor
  efficiency), signed: positive = draw, negative = regen recovery.
* ``hvac_wh_tick``   — climate load, only while a trip defines a cabin set
  point (simulation parameter profile, NOT measured TUM data).
* ``aux_wh_tick``    — constant auxiliary base load.

``battery_wh_tick`` is the ONLY quantity used to drain SOC; HVAC is folded in
here and nowhere else, so it is never double-counted. All constants come from
the versioned ``simparams`` profile (sourced vs assumed provenance).
"""

from __future__ import annotations

import math
import random
import time

from .scenarios import ScenarioState, ScenarioType
from .simparams import DEFAULT_PROFILE as _P

# Module-level constants are kept for back-compat but all values are sourced
# from the versioned profile so provenance is explicit (sourced vs assumed).
G = _P.value("g_mps2")
MASS_KG = _P.value("mass_kg")
FRONTAL_AREA_M2 = _P.value("frontal_area_m2")
CD = _P.value("cd")
RHO_AIR = _P.value("rho_air_kgm3")
CRR = _P.value("crr")
MAX_MOTOR_W = _P.value("max_motor_w")
MAX_REGEN_W = _P.value("max_regen_w")
MAX_ENGINE_W = _P.value("max_engine_w")
BATTERY_WH = _P.value("battery_usable_wh")
NOMINAL_TIRE_KPA = 240.0
HVAC_COEFF_W_PER_C = _P.value("hvac_coeff_w_per_c")
HVAC_BASE_W = _P.value("hvac_base_w")
HVAC_MIN_W = _P.value("hvac_min_w")
HVAC_MAX_W = _P.value("hvac_max_w")
AUX_BASE_W = _P.value("aux_base_w")


class VehicleModel:
    """Advances the vehicle one tick at a time."""

    def __init__(self, vehicle_id: str, seed: int | None = None) -> None:
        self.vehicle_id = vehicle_id
        self.rng = random.Random(seed)
        self.seq = 0
        self.speed = 0.0  # kph
        self.accel = 0.0  # m/s^2
        self.heading = self.rng.uniform(0.0, 360.0)
        self.lat = 47.4502
        self.lon = -122.3088
        self.alt = 120.0
        self.battery_soc = self.rng.uniform(62.0, 88.0)
        self._depleted = False
        self.battery_temp = 24.0
        self.coolant_temp = 82.0
        self.tire = {k: NOMINAL_TIRE_KPA for k in ("fl", "fr", "rl", "rr")}
        self.wh_per_km: list[float] = [self.rng.uniform(190.0, 250.0) for _ in range(40)]
        self.mode = "eco"
        self.charging = False
        self.dtc_codes: list[str] = []
        self.dtc_frame_counter: dict[str, int] = {}
        self.target_speed = 0.0
        self.phase_remaining = 0.0
        self.charging_timer = 0.0
        self.scenario = ScenarioState()
        self._wh_this_tick = 0.0
        self._distance_this_tick = 0.0
        self._session_wh = 0.0
        self._session_km = 0.0
        self.odometer_km = self.rng.uniform(11800.0, 14200.0)
        # WS1 trip state
        self._trip_seg = 0
        self._trip_phase_s = 0.0
        self._trip_arrival_handled = False
        # Live driver override (cruise). None == follow the active plan
        # (trip speed profile, or the free-drive phase chooser).
        self._override_speed_kph: float | None = None
        # WS1 single-only energy ledger (see module docstring)
        self._motive_wh_tick = 0.0
        self._hvac_wh_tick = 0.0
        self._aux_wh_tick = 0.0
        self._battery_wh_tick = 0.0
    # -- public API ----------------------------------------------------------

    def configure(self, scenario_type: ScenarioType, duration_s: float, intensity: float) -> None:
        self.scenario.activate(scenario_type, duration_s, intensity)
        self._trip_arrival_handled = False
        if scenario_type not in (ScenarioType.CHARGE, ScenarioType.CHARGE_STALL):
            # Leaving the charging yard re-arms driving: a direct hand-over like
            # charge_stall -> battery_heat must not stay parked on a stale timer.
            self.charging = False
        if self.scenario.soc_target_pct is not None:
            # LOW_ENERGY floor: clamp the pack so the condition is real & reachable.
            self.battery_soc = min(self.battery_soc, self.scenario.soc_target_pct)

    def configure_trip(
        self,
        destination_km: float,
        cabin_set_temp_c: float,
        ambient_temp_c: float,
        speed_profile_kph: list[float] | None = None,
        segment_s: float = 120.0,
    ) -> None:
        """Activate a destination-aware causal trip (SCENARIO mode)."""
        self.scenario.activate_trip(
            destination_km,
            cabin_set_temp_c,
            ambient_temp_c,
            speed_profile_kph=speed_profile_kph,
            segment_s=segment_s,
        )
        self._trip_seg = 0
        self._trip_phase_s = 0.0
        self._trip_arrival_handled = False
        self._override_speed_kph = None  # a fresh plan; clear any driver cruise override
        self.target_speed = self.scenario.trip.speed_profile_kph[0]

    def set_target_speed(self, value: float | None) -> float | None:
        """Live cruise command (driver action). Follow the plan when None."""
        self._override_speed_kph = None if value is None else max(0.0, float(value))
        return self._override_speed_kph

    def set_cabin_set_temp(self, value: float | None) -> float | None:
        """Update the set point of an active trip (HVAC exists only in a trip).

        Returns the accepted set point, or None when there is no active trip —
        WS1: never fabricate an HVAC set point for non-trip telemetry.
        """
        trip = self.scenario.trip
        if value is None or trip is None:
            return None
        trip.cabin_set_temp_c = float(min(30.0, max(12.0, value)))
        return trip.cabin_set_temp_c

    def control_status(self) -> dict:
        """State of the live driver-controlled surfaces (not telemetry signals)."""
        trip = self.scenario.trip
        hvac_w, _ = self._aux_power_w() if trip is not None else (0.0, 0.0)
        return {
            "mode": self.scenario.mode,
            "scenario": self.scenario.active_type.value,
            "charging": self.charging,
            "target_speed_kph": round(self._override_speed_kph, 1)
            if self._override_speed_kph is not None
            else round(self.target_speed, 1),
            "override_speed_kph": round(self._override_speed_kph, 1) if self._override_speed_kph is not None else None,
            "cabin_set_temp_c": round(trip.cabin_set_temp_c, 1) if trip is not None else None,
            "ambient_temp_c": round(trip.ambient_temp_c, 1) if trip is not None else None,
            "hvac_power_kw": round(hvac_w / 1000.0, 3) if trip is not None else 0.0,
            "trip_active": trip is not None,
            "destination_km": round(trip.destination_km, 1) if trip is not None else None,
            "remaining_km": round(trip.remaining_km, 2) if trip is not None and not trip.completed else None,
            "charge_rate_kw": round(self.scenario.charge_rate_kw, 2) if self.charging else 0.0,
        }

    def tick(self, dt: float) -> dict:
        self.seq += 1
        self._wh_this_tick = 0.0
        self._distance_this_tick = 0.0
        self._motive_wh_tick = 0.0
        self._hvac_wh_tick = 0.0
        self._aux_wh_tick = 0.0
        self._battery_wh_tick = 0.0
        self._advance_drive_phase(dt)
        trac_w = self._compute_traction_power(dt)
        if self.charging:
            trac_w = 0.0
            self._advance_charging(dt)

        engine_share = self._battery_and_temps(trac_w, dt)
        self._advance_tires(dt)
        self._advance_gps(dt)
        self._advance_dtcs(dt)
        self._handle_trip_arrival(dt)
        warnings = self._warnings()
        dist_km = self.speed / 3.6 * dt / 1000.0
        self._session_km += dist_km
        self._session_wh += abs(self._wh_this_tick)
        self.odometer_km += dist_km
        self.scenario.tick(dt)
        return self._frame(trac_w, engine_share, warnings)

    # -- internals -----------------------------------------------------------

    def _advance_drive_phase(self, dt: float) -> None:
        # Once depleted, remain parked until recharged; regeneration must not
        # make an empty battery alternate between acceleration and braking.
        if self.battery_soc <= 0.05:
            self._depleted = True
        if self._depleted and self.battery_soc >= 1.0:
            self._depleted = False
        if self._depleted and not self.charging and self.scenario.active_type not in (ScenarioType.CHARGE, ScenarioType.CHARGE_STALL):
            old_speed = self.speed
            self.speed = max(0.0, self.speed - 1.6 * dt * 3.6)
            self.accel = (self.speed - old_speed) / (3.6 * max(dt, 1e-6))
            self.target_speed = 0.0
            return
        trip = self.scenario.trip
        if self.scenario.active_type == ScenarioType.TRIP and trip is not None:
            self._advance_trip_phase(dt, trip)
            return
        if (
            self._trip_arrival_handled
            and self.scenario.active_type == ScenarioType.NONE
            and not self.charging
        ):
            # Post-trip: park at the arrival point instead of resuming random
            # driving. A new scenario / trip re-arms driving.
            self.speed = max(0.0, self.speed - 4.2 * dt * 3.6)
            self.accel = -4.2
            self.target_speed = 0.0
            return
        stop = False
        if self.charging:
            self.charging_timer -= dt
            if self.charging_timer <= 0:
                self.charging = False
            self.speed = 0.0
            self.accel = 0.0
            return
        if self.scenario.active_type in (ScenarioType.CHARGE, ScenarioType.CHARGE_STALL):
            self.charging = True
            self.charging_timer = self.scenario.remaining_s
            return

        self.phase_remaining -= dt
        if self.phase_remaining <= 0:
            self._choose_drive_phase()
        # Driver: accelerate toward target with per-mode aggressiveness.
        if self._override_speed_kph is not None:
            self.target_speed = self._override_speed_kph
        dv = (self._target_now() - self.speed) / 3.6
        max_a = {"eco": 1.6, "normal": 2.4, "sport": 3.4}[self.mode]
        max_dec = 4.2
        a = max(-max_dec, min(max_a, dv / max(dt, 1e-6)))
        a = max(max_a * -1.0, min(max_a, a * 0.3 + 0.7 * a))  # realism filter
        self.accel = a
        self.speed = max(0.0, self.speed + a * dt * 3.6)
        # speed-controlled cornering drift on straights
        if self.seq % 37 == 0 and self.rng.random() < 0.3:
            self.heading = (self.heading + self.rng.uniform(-3, 3)) % 360.0

    def _advance_trip_phase(self, dt: float, trip) -> None:
        """Drive the vehicle along the requested speed profile (trip mode)."""
        profile = trip.speed_profile_kph
        if self._trip_arrival_handled:
            # Arrival already processed: glide to a stop, then end the trip.
            self.speed = max(0.0, self.speed - 4.2 * dt * 3.6)
            self.accel = -4.2
            self.target_speed = 0.0
            if self.speed <= 0.5 and self.scenario.active_type == ScenarioType.TRIP:
                self.scenario.end_trip()
            return
        self._trip_phase_s += dt
        if self._trip_phase_s >= trip.segment_s:
            self._trip_phase_s = 0.0
            self._trip_seg = min(self._trip_seg + 1, len(profile) - 1)
        target = (
            self._override_speed_kph
            if self._override_speed_kph is not None
            else max(0.0, profile[self._trip_seg])
        )
        self.target_speed = target
        self.mode = "eco"
        dv = (target - self.speed) / 3.6
        max_a = 1.6
        max_dec = 4.2
        a = max(-max_dec, min(max_a, dv / max(dt, 1e-6)))
        self.accel = a
        self.speed = max(0.0, self.speed + a * dt * 3.6)
        if self.seq % 37 == 0 and self.rng.random() < 0.3:
            self.heading = (self.heading + self.rng.uniform(-3, 3)) % 360.0

    def _choose_drive_phase(self) -> None:
        drift = self.scenario.speed_bias_kph
        r = self.rng.random()
        if r < 0.42:
            self.target_speed, self.phase_remaining, self.mode = (
                self.rng.uniform(30, 58) + drift,
                self.rng.uniform(12, 40),
                "eco",
            )
        elif r < 0.78:
            self.target_speed, self.phase_remaining, self.mode = (
                self.rng.uniform(90, 118) + drift,
                self.rng.uniform(25, 70),
                "normal",
            )
        else:
            self.target_speed, self.phase_remaining, self.mode = (
                self.rng.uniform(120, 138) + drift,
                self.rng.uniform(15, 45),
                "sport",
            )
        self.target_speed = max(0.0, self.target_speed)

    def _target_now(self) -> float:
        drift = self.scenario.speed_bias_kph
        wobble = self.rng.uniform(-2.0, 2.0)
        target = max(0.0, self.target_speed + drift + wobble)
        # stop-start traffic pulses
        if self.scenario.active_type == ScenarioType.DRIFT:
            target = min(target, 40.0 + self.scenario.speed_bias_kph)
        return target

    def _compute_traction_power(self, dt: float) -> float:
        v_ms = self.speed / 3.6
        if v_ms < 0.05:
            self.speed = 0.0
            self.accel = 0.0
            return 0.0
        p_roll = MASS_KG * G * CRR * v_ms
        p_aero = 0.5 * RHO_AIR * CD * FRONTAL_AREA_M2 * v_ms**3
        p_inertia = MASS_KG * self.accel * v_ms
        demand = p_roll + p_aero + p_inertia
        if demand >= 0:
            load = min(1.0, abs(demand) / MAX_MOTOR_W)
            eta = 0.70 + 0.22 * math.sqrt(max(load, 0.05))
            self._wh_this_tick = demand * dt / 3600.0
            return demand / min(eta, 0.94)
        regen_cap = MAX_REGEN_W
        recover = min(-demand * 0.78, regen_cap)
        self._wh_this_tick = -recover * dt / 3600.0
        return -recover

    def _battery_and_temps(self, p_w: float, dt: float) -> float:
        engine_share = 0.0
        # This cockpit represents a BEV: no hidden combustion-engine assist.
        motive = p_w - engine_share
        if self.mode == "sport" and p_w > 0:
            motive = p_w  # engine assist allowed

        # ---- WS1 single energy ledger: motive + hvac + aux, summed exactly once.
        motive_wh = motive * dt / 3600.0
        hvac_w, aux_w = self._aux_power_w()
        hvac_wh = hvac_w * dt / 3600.0
        aux_wh = (aux_w if not self.charging else 0.0) * dt / 3600.0
        battery_wh = motive_wh + hvac_wh + aux_wh
        self._motive_wh_tick = motive_wh
        self._hvac_wh_tick = hvac_wh
        self._aux_wh_tick = aux_wh
        self._battery_wh_tick = battery_wh

        eff_base = self.scenario.battery_efficiency_factor
        self.battery_soc = min(100.0, max(0.0, self.battery_soc - battery_wh / BATTERY_WH * 100.0 / eff_base))
        heat = (abs(motive) * 0.055 + 480.0) * dt
        b_temp_target = 23.0 + heat / 30.0 + self.scenario.battery_temp_bias
        self.battery_temp += ((b_temp_target - self.battery_temp) * 0.06) + self.rng.uniform(-0.15, 0.15)
        cool_target = 88.0 + (abs(motive) / 165_000.0) * 34.0
        self.coolant_temp += ((cool_target - self.coolant_temp) * 0.12) + self.rng.uniform(-0.1, 0.1)
        self.coolant_temp = max(20.0, self.coolant_temp)
        self._distance_this_tick = self.speed / 3.6 * dt / 1000.0
        return engine_share

    def _aux_power_w(self) -> tuple[float, float]:
        """(hvac_w, aux_w) — simulation loads, NOT measured TUM data.

        HVAC load only exists while a trip defines a cabin set point. Without a
        trip there is no cabin set point, so HVAC is zero (as requested: never
        fabricate HVAC for non-trip / replayed telemetry).
        """
        aux_w = AUX_BASE_W
        trip = self.scenario.trip
        hvac_w = 0.0
        if trip is not None:
            delta_c = trip.cabin_set_temp_c - trip.ambient_temp_c
            raw = HVAC_COEFF_W_PER_C * delta_c + HVAC_BASE_W
            hvac_w = min(HVAC_MAX_W, max(HVAC_MIN_W, raw))
        return hvac_w, aux_w

    def _advance_charging(self, dt: float) -> None:
        rate = self.scenario.charge_rate_kw  # 11 kW nominal, 0 kW on a charge stall
        self.battery_soc = min(100.0, self.battery_soc + rate * dt / 3600.0 / (BATTERY_WH / 1000.0) * 100.0)
        self.coolant_temp += (35.0 - self.coolant_temp) * 0.03
        self.battery_temp += (26.0 - self.battery_temp) * 0.04

    def _advance_tires(self, dt: float) -> None:
        speed_effect = 1.2 * (self.speed / 100.0)
        wobble = self.rng.uniform(-0.8, 0.8)
        for k in self.tire:
            base = NOMINAL_TIRE_KPA + speed_effect
            self.tire[k] = self.tire[k] + (base - self.tire[k]) * 0.005 + wobble
        leak = self.scenario.leaking_wheel
        if leak:
            self.tire[leak] -= self.scenario.leak_rate_kpa * dt

    def _advance_gps(self, dt: float) -> None:
        v_ms = self.speed / 3.6
        if v_ms < 0.05:
            return
        d_km = v_ms * dt / 1000.0
        # ~111.32 km per degree
        self.lat += (d_km * math.cos(math.radians(self.heading))) / 111.32
        self.lon += (d_km * math.sin(math.radians(self.heading))) / (111.32 * math.cos(math.radians(self.lat)))
        self.alt += self.rng.uniform(-0.3, 0.3)
        trip = self.scenario.trip
        if trip is not None:
            trip.remaining_km = max(0.0, trip.remaining_km - d_km)

    def _handle_trip_arrival(self, dt: float) -> None:
        """Trigger the arrival glide when the destination is reached.

        The trip object stays active until the vehicle comes to a stop; the
        glide branch in ``_advance_trip_phase`` calls ``scenario.end_trip()``.
        """
        if self.scenario.active_type != ScenarioType.TRIP or self.scenario.trip is None:
            return
        trip = self.scenario.trip
        if trip.remaining_km <= 0.0 and not self._trip_arrival_handled:
            self._trip_arrival_handled = True
            self.target_speed = 0.0

    def _advance_dtcs(self, dt: float) -> None:
        proposal = ["P0420", "U0100", "B0131", "P0171", "C0031"]
        prob = {"P0420": 0.000045, "U0100": 0.00002, "B0131": 0.000012, "P0171": 0.00003, "C0031": 0.00005}
        for code in proposal:
            if code in self.dtc_codes:
                continue
            p = prob[code] * (1.0 + self.scenario.dtc_bias)
            if self.rng.random() < p:
                self.dtc_codes.append(code)
                self.dtc_frame_counter[code] = 0
        for code in list(self.dtc_codes):
            self.dtc_frame_counter[code] = self.dtc_frame_counter.get(code, 0) + 1
            # DTCs self-clear after a couple of minutes unless fault persists.
            if self.dtc_frame_counter[code] > 1200 and self.rng.random() < 0.01:
                self.dtc_codes.remove(code)

    def _warnings(self) -> list[dict]:
        out: list[dict] = []
        for k in ("fl", "fr", "rl", "rr"):
            if self.tire[k] < 200:
                out.append({"kind": "tire_pressure_low", "message": f"Low tire pressure: {k} @ {self.tire[k]:.0f} kPa", "severity": "warning"})
        if self.battery_temp > 60:
            # Only a genuinely thermal HV-battery event is critical; an elevated
            # pack (42-60 C) is a warning, never an immediate-stop condition.
            out.append(
                {
                    "kind": "battery_overtemp",
                    "message": f"Battery overheating — thermal risk: {self.battery_temp:.1f} C",
                    "severity": "critical",
                }
            )
        elif self.battery_temp > 42:
            out.append(
                {
                    "kind": "battery_overtemp",
                    "message": f"Battery temperature elevated: {self.battery_temp:.1f} C",
                    "severity": "warning",
                }
            )
        if self.coolant_temp > 103:
            out.append({"kind": "coolant_overtemp", "message": f"Coolant over temperature: {self.coolant_temp:.1f} C", "severity": "warning"})
        if 10 <= self.battery_soc < 15:
            out.append({"kind": "low_soc", "message": "Charge level critical (10-15%)", "severity": "warning"})
        if self.battery_soc < 10:
            out.append({"kind": "low_soc", "message": "Battery below 10 percent; remaining range is limited", "severity": "critical"})
        return out

    def _frame(self, trac_w: float, engine_share: float, warnings: list[dict]) -> dict:
        v = self.speed
        wh_per_km = 0.0
        if self._distance_this_tick > 1e-6 and self._wh_this_tick:
            # Instantaneous Wh/km is meaningless during parking-lot creep and
            # spikes when regen hand-off happens; only learn from real motion,
            # clamped so the range estimator stays sane.
            if v >= 9.0:
                ratio = abs(self._wh_this_tick) / self._distance_this_tick
                wh_per_km = min(max(ratio, 60.0), 520.0)
        if wh_per_km > 0:
            self.wh_per_km.append(wh_per_km)
            self.wh_per_km = self.wh_per_km[-400:]
        avg_wh_km = sum(self.wh_per_km) / len(self.wh_per_km) if self.wh_per_km else 220.0
        dte = self.battery_soc / 100.0 * BATTERY_WH / max(avg_wh_km, 1.0)

        charge_rate = self.scenario.charge_rate_kw if self.charging else 0.0
        soc_bias = self.scenario.soc_bias
        trip = self.scenario.trip
        hvac_w, aux_w = self._aux_power_w()
        ambient = round(trip.ambient_temp_c, 1) if trip is not None else round(18.0 + self.rng.uniform(-1, 1), 1)
        arrival_eta_min = None
        if trip is not None and v > 1.0:
            arrival_eta_min = round(trip.remaining_km / v * 60.0, 1)
        signals = {
            "vehicle_speed_kph": round(v, 1),
            "accel_mps2": round(self.accel, 2),
            "engine_rpm": round((self.speed / 3.6 / 3.0) * 60 * 8 * (1 if self.mode == "sport" else 0.5)) if not self.charging else 0,
            "motor_power_kw": round(-trac_w / 1000.0, 1),
            "engine_power_kw": round(engine_share / 1000.0, 1),
            "regen_power_kw": round(min(0.0, trac_w) / 1000.0, 1),
            "battery_soc_pct": round(min(100.0, max(0.0, self.battery_soc + soc_bias)), 1),
            "battery_temp_c": round(self.battery_temp, 1),
            "coolant_temp_c": round(self.coolant_temp, 1),
            "outside_temp_c": ambient,
            "cabin_temp_c": round(21.5, 1),
            "cabin_humidity_pct": round(self.rng.uniform(35, 52), 1),
            "fuel_level_pct": round(1.0 + self.scenario.fuel_level_pct, 1) if not self.charging else 1.0,
            "charge_rate_kw": charge_rate,
            "distance_to_empty_km": round(dte, 1),
            "energy_session_kwh": round(self.wh_this_session / 3600.0, 2),
            "trip_distance_km": round(self.trip_km, 2),
            "accelerator_pedal_pct": round(self.rng.uniform(2, 60) if v > 5 else 0.0, 1),
            "brake_pressure_pct": round(self.rng.uniform(0, 8) if self.accel < -0.5 else 0.0, 1),
            "tire_pressure_fl_kpa": round(self.tire["fl"], 1),
            "tire_pressure_fr_kpa": round(self.tire["fr"], 1),
            "tire_pressure_rl_kpa": round(self.tire["rl"], 1),
            "tire_pressure_rr_kpa": round(self.tire["rr"], 1),
            "battery_voltage_v": round(355.0 - (100.0 - self.battery_soc) * 0.42, 1),
            "pack_current_a": round((trac_w / 1000.0) / 355.0 * 1000.0, 1),
            "cabin_set_temp_c": round(trip.cabin_set_temp_c, 1) if trip is not None else round(self.rng.uniform(20.0, 23.0), 1),
            "odometer_km": round(self.odometer_km, 1),
            "charge_port_latched": 1.0 if self.charging else 0.0,
            # ---- WS1 trip + energy-ledger signals (additive; schema unchanged)
            "destination_remaining_km": -1.0 if trip is None else round(trip.remaining_km, 2),
            "hvac_power_kw": round(hvac_w / 1000.0, 3),
            "ambient_temp_c": ambient,
            "arrival_eta_min": -1.0 if arrival_eta_min is None else arrival_eta_min,
            "motive_energy_wh_tick": round(self._motive_wh_tick, 3),
            "hvac_energy_wh_tick": round(self._hvac_wh_tick, 3),
            "aux_energy_wh_tick": round(self._aux_wh_tick, 3),
            "battery_energy_wh_tick": round(self._battery_wh_tick, 3),
        }
        if "U0100" in self.dtc_codes:  # lost comm with brake module -> signal dropouts
            signals["brake_pressure_pct"] = float("nan")
        self._wh_this_tick = 0.0
        self._distance_this_tick = 0.0
        return {
            "vehicle_id": self.vehicle_id,
            "timestamp": time.time(),
            "seq": self.seq,
            "driving_mode": self.mode,
            "charging": self.charging,
            "scenario_mode": self.scenario.mode,
            "signals": signals,
            "active_dtcs": self.dtc_codes[:],
            "warnings": warnings,
            "gps": {
                "lat": round(self.lat, 6),
                "lon": round(self.lon, 6),
                "altitude_m": round(self.alt, 1),
                "heading_deg": round(self.heading, 1),
            },
        }

    @property
    def wh_this_session(self) -> float:
        return self._session_wh

    @property
    def trip_km(self) -> float:
        return self._session_km

    # -- WS1 energy ledger (motive + hvac + aux summed exactly once) ----------

    @property
    def motive_wh_tick(self) -> float:
        return self._motive_wh_tick

    @property
    def hvac_wh_tick(self) -> float:
        return self._hvac_wh_tick

    @property
    def aux_wh_tick(self) -> float:
        return self._aux_wh_tick

    @property
    def battery_wh_tick(self) -> float:
        return self._battery_wh_tick

    @property
    def energy_ledger(self) -> dict:
        """Per-tick battery energy accounting: motive + hvac + aux = total."""
        return {
            "motive_wh_tick": self._motive_wh_tick,
            "hvac_wh_tick": self._hvac_wh_tick,
            "aux_wh_tick": self._aux_wh_tick,
            "battery_wh_tick": self._battery_wh_tick,
        }

    @property
    def trip(self):
        return self.scenario.trip