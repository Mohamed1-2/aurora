"""Versioned vehicle / simulator parameter profile with provenance.

Every constant used by the physics-lite simulator and the WS1 energy engine
lives here. Each parameter carries:

* ``parameter`` — stable machine name
* ``value``     — numeric value in ``unit``
* ``unit``      — SI-ish unit of ``value``
* ``provenance``— where the number came from ("simulation", "public_spec", ...)
* ``source_reference`` — specific reference if available, else None
* ``assumption``— True when the value is an engineering/simulation assumption
                  and must NOT be presented as a measured TUM value or exact
                  VW/CUPRA engineering parameter.

Nothing in this module implies these are measured TUM values or manufacturer
guaranteed parameters. ``assumption=True`` values are deliberately flagged so
the assistant / evaluation harness can distinguish *sourced* from *assumed*.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimParameter:
    parameter: str
    value: float
    unit: str
    provenance: str
    source_reference: str | None = None
    assumption: bool = True
    notes: str = ""


@dataclass(frozen=True)
class SimulatorParameterProfile:
    name: str
    version: str
    vehicle_class: str
    parameters: dict[str, SimParameter]

    def get(self, key: str) -> SimParameter:
        return self.parameters[key]

    def value(self, key: str) -> float:
        return self.parameters[key].value

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "vehicle_class": self.vehicle_class,
            "parameters": {
                k: {
                    "parameter": p.parameter,
                    "value": p.value,
                    "unit": p.unit,
                    "provenance": p.provenance,
                    "source_reference": p.source_reference,
                    "assumption": p.assumption,
                    "notes": p.notes,
                }
                for k, p in sorted(self.parameters.items())
            },
        }

    def sourced(self) -> list[SimParameter]:
        return [p for p in self.parameters.values() if not p.assumption]

    def assumed(self) -> list[SimParameter]:
        return [p for p in self.parameters.values() if p.assumption]


def _sim(name: str, value: float, unit: str, *, notes: str = "", source: str | None = None) -> SimParameter:
    """A simulation assumption: not measured, not manufacturer-verified."""
    return SimParameter(
        parameter=name,
        value=value,
        unit=unit,
        provenance="simulation_assumption",
        source_reference=source,
        assumption=True,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# WS1 profile. Values are simulation parameters for the controlled synthetic
# scenario simulator; they are NOT claimed to be measured TUM telemetry or
# exact VW/CUPRA engineering data (see module docstring).
# ---------------------------------------------------------------------------
V1 = SimulatorParameterProfile(
    name="aurora-sim-params",
    version="ws1-v1",
    vehicle_class="Cupra-Born-like-BEV (simulated)",
    parameters={
        # --- environment / physics ---
        "g_mps2": _sim("g_mps2", 9.81, "m/s^2", notes="standard gravity"),
        "rho_air_kgm3": _sim("rho_air_kgm3", 1.225, "kg/m^3", notes="sea-level air density, 15 C"),
        "mass_kg": _sim("mass_kg", 2100.0, "kg", notes="approx curb+load"),
        "frontal_area_m2": _sim("frontal_area_m2", 2.62, "m^2"),
        "cd": _sim("cd", 0.24, "dimensionless", notes="aerodynamic drag coefficient"),
        "crr": _sim("crr", 0.011, "dimensionless", notes="rolling resistance coefficient"),
        # --- drivetrain / battery ---
        "max_motor_w": _sim("max_motor_w", 165_000.0, "W"),
        "max_regen_w": _sim("max_regen_w", 55_000.0, "W"),
        "max_engine_w": _sim("max_engine_w", 92_000.0, "W", notes="range extender (only used <18% SOC)"),
        "battery_usable_wh": _sim("battery_usable_wh", 82_000.0, "Wh", notes="usable battery capacity, NOT manufacturer sticker"),
        "regen_recovery_factor": _sim("regen_recovery_factor", 0.78, "dimensionless"),
        # --- HVAC (simulation load model, NOT measured TUM data) ---
        "hvac_coeff_w_per_c": _sim("hvac_coeff_w_per_c", 220.0, "W/C", notes="heating/cooling power per cabin-ambient delta"),
        "hvac_base_w": _sim("hvac_base_w", 400.0, "W", notes="blower + electronics floor load"),
        "hvac_min_w": _sim("hvac_min_w", 300.0, "W"),
        "hvac_max_w": _sim("hvac_max_w", 5000.0, "W"),
        # --- auxiliary load (non-HVAC) ---
        "aux_base_w": _sim("aux_base_w", 350.0, "W", notes="12V bus + infotainment + lighting base draw"),
    },
)

PROFILES: dict[str, SimulatorParameterProfile] = {"ws1-v1": V1}
DEFAULT_PROFILE: SimulatorParameterProfile = V1


def profile(version: str = "ws1-v1") -> SimulatorParameterProfile:
    try:
        return PROFILES[version]
    except KeyError:
        raise KeyError(f"unknown simulator parameter profile version={version!r}") from None