"""Scenario injection: how the ops team forces interesting states on purpose.

Used by the demo tool and by chaos-drill automation to prove that the
assistant + drift monitors react to real change, not just look busy.

Two explicit, non-blended modes are supported (WS1):

* ``mode="scenario"``  — synthetic/controlled causal trip (``TripState``)
* ``mode="tum_replay"`` — genuine measured TUM telemetry replayed from raw
  parquet (handled by ``AppState``/``tum_replay``; nothing synthetic here).

``ScenarioState`` carries only the scenario mode. The replay feed never mixes
with the synthetic scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScenarioType(str, Enum):
    NONE = "none"
    LEAK = "leak"          # slow tire pressure leak on one wheel
    DTC = "dtc"            # elevation of diagnostic trouble codes
    DRIFT = "drift"        # biased speed/SOC distribution -> statistical drift
    CHARGE = "charge"      # vehicle parks and charges
    CHARGE_STALL = "charge_stall"  # plugged in, but charging power ~ 0 (fault)
    BATTERY_HEAT = "battery_heat"  # battery cooling degraded -> overtemperature
    LOW_ENERGY = "low_energy"      # pre-drained pack: long-range risk arrival trip
    TRIP = "trip"          # destination-aware causal trip (WS1)


@dataclass
class TripState:
    destination_km: float
    cabin_set_temp_c: float
    ambient_temp_c: float
    speed_profile_kph: list[float] = field(default_factory=lambda: [60.0, 90.0, 110.0])
    segment_s: float = 120.0
    remaining_km: float = 0.0
    completed: bool = False

    def __post_init__(self) -> None:
        self.destination_km = max(1.0, self.destination_km)
        if self.remaining_km <= 0.0:
            self.remaining_km = self.destination_km


class ScenarioState:
    def __init__(self) -> None:
        self.active_type: ScenarioType = ScenarioType.NONE
        self.remaining_s = 0.0
        self.speed_bias_kph = 0.0
        self.soc_bias = 0.0
        self.battery_temp_bias = 0.0
        self.battery_efficiency_factor = 1.0
        self.dtc_bias = 0.0
        self.leaking_wheel: str | None = None
        self.leak_rate_kpa = 0.0
        self.fuel_level_pct = 0.0
        self.charge_rate_kw = 11.0          # kW while parked & charging (stall => 0)
        self.soc_target_pct: float | None = None  # low_energy: drain pack to this floor
        self.mode: str = "scenario"
        self.trip: TripState | None = None

    @property
    def label(self) -> str:
        return self.active_type.value

    def activate(self, scenario_type: str, duration_s: float, intensity: float) -> None:
        self.reset()
        self.active_type = ScenarioType(scenario_type)
        self.remaining_s = max(1.0, duration_s)
        i = min(max(intensity, 0.0), 1.0)
        if self.active_type == ScenarioType.LEAK:
            self.leaking_wheel = "rr"
            self.leak_rate_kpa = 0.35 + 1.4 * i  # kPa per second
        elif self.active_type == ScenarioType.DTC:
            self.dtc_bias = 20.0 * i
        elif self.active_type == ScenarioType.DRIFT:
            self.speed_bias_kph = 45.0 * i
            self.soc_bias = -12.0 * i   # shift SOC distribution down
        elif self.active_type == ScenarioType.CHARGE:
            self.charge_rate_kw = 11.0   # nominal AC L2 charge
        elif self.active_type == ScenarioType.CHARGE_STALL:
            self.charge_rate_kw = 0.0    # plugged in -> power never arrives (fault)
        elif self.active_type == ScenarioType.BATTERY_HEAT:
            # Enough bias to cross the sim's 60 C critical/thermal boundary only
            # at intensity 1 (demo intent); lower intensities stay a warning.
            self.battery_temp_bias = 40.0 * i
            self.battery_efficiency_factor = 1.0 - 0.12 * i
        elif self.active_type == ScenarioType.LOW_ENERGY:
            # Pre-drained pack floor: the model clamps SOC to this target when
            # the scenario activates, so the "low arrival energy" condition is
            # reachable in seconds instead of after a simulated hour of driving.
            self.soc_target_pct = 14.5

    def activate_trip(
        self,
        destination_km: float,
        cabin_set_temp_c: float,
        ambient_temp_c: float,
        speed_profile_kph: list[float] | None = None,
        segment_s: float = 120.0,
    ) -> None:
        """Activate a destination-aware causal trip (SCENARIO mode)."""
        if speed_profile_kph is None:
            speed_profile_kph = [60.0, 90.0, 110.0]
        self.reset()
        self.active_type = ScenarioType.TRIP
        self.mode = "scenario"
        self.remaining_s = float("inf")  # TRIP self-terminates on arrival, not timer
        self.trip = TripState(
            destination_km=max(1.0, destination_km),
            cabin_set_temp_c=cabin_set_temp_c,
            ambient_temp_c=ambient_temp_c,
            speed_profile_kph=[max(0.0, s) for s in speed_profile_kph],
            segment_s=max(1.0, segment_s),
            remaining_km=max(1.0, destination_km),
        )

    def set_replay_mode(self) -> None:
        """Mark the feed as genuine TUM replayed telemetry (NON-synthetic)."""
        self.reset()
        self.mode = "tum_replay"

    def tick(self, dt: float) -> None:
        if self.active_type != ScenarioType.NONE and self.active_type != ScenarioType.TRIP:
            self.remaining_s -= dt
            if self.remaining_s <= 0:
                self.reset()

    def end_trip(self) -> None:
        """Called by the model when a trip reaches its destination."""
        if self.trip is not None:
            self.trip.completed = True
        self.active_type = ScenarioType.NONE
        self.trip = None

    def reset(self) -> None:
        self.__init__()


SCENARIOS = sorted({e.value for e in ScenarioType})