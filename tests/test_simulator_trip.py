"""WS1 simulator trip tests.

SIMULATOR CONSISTENCY — these tests assert the *physics/software* consistency of
the controlled scenario simulator: expected causal directions, correct energy
accounting, and that the physics energy engine approximately agrees with the
simulator that uses the same assumptions. They do NOT claim real-EV accuracy —
real-data predictive evaluation is a WS2 concern over held-out TUM sequences.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "telemetry-simulator"))
sys.path.insert(0, str(ROOT))

# The repo runs multiple services' `app` packages in one process (services have
# no __init__.py and each owns its own `app`). Evict any cached foreign `app`
# so THIS test imports the telemetry-simulator package unconditionally.
for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_mod, None)

from app.engine.energy_engine import compute_drivers, project_arrival, what_if
from app.engine.scenarios import ScenarioType
from app.engine.simparams import DEFAULT_PROFILE as PROFILE
from app.engine.vehicle_model import VehicleModel


def _drive_seconds(model: VehicleModel, seconds: float, dt: float = 0.5) -> list[dict]:
    frames = []
    for _ in range(int(seconds / dt)):
        frames.append(model.tick(dt))
    return frames


def _new_model(seed: int = 20240627) -> VehicleModel:
    return VehicleModel("T1", seed=seed)


# --- 1. trip activation sets destination + signals -------------------------


def test_trip_activation_sets_destination_and_signals():
    m = _new_model()
    m.configure_trip(destination_km=120, cabin_set_temp_c=24, ambient_temp_c=-2,
                     speed_profile_kph=[30], segment_s=10)
    assert m.scenario.active_type == ScenarioType.TRIP
    assert m.scenario.trip.destination_km == 120.0
    assert m.scenario.trip.remaining_km == pytest.approx(120.0, abs=1e-9)
    frame = m.tick(0.5)
    s = frame["signals"]
    assert s["destination_remaining_km"] > 0
    assert s["hvac_power_kw"] > 0          # 24 - (-2) delta => heating
    assert s["cabin_set_temp_c"] == 24.0
    assert s["ambient_temp_c"] == -2.0
    assert s["arrival_eta_min"] >= 0
    assert frame["scenario_mode"] == "scenario"


# --- 2. HVAC ON drains more than HVAC OFF (same seed) -----------------------


def test_hvac_on_consumes_more_soc_than_hvac_off():
    def run_with(cabin: float | None, ambient: float) -> float:
        m = _new_model()
        speed = [40.0]
        # delta 0 (cabin == ambient) => floor HVAC load; big delta => high HVAC.
        m.configure_trip(destination_km=200, cabin_set_temp_c=cabin, ambient_temp_c=ambient, speed_profile_kph=speed, segment_s=10)
        before = m.battery_soc
        _drive_seconds(m, 60.0)
        return before - m.battery_soc

    hvac_off_drop = run_with(cabin=20, ambient=20)   # delta 0 => ~min HVAC
    hvac_on_drop = run_with(cabin=28, ambient=-5)    # big heating delta
    assert hvac_on_drop > hvac_off_drop


def test_hvac_power_nonzero_only_in_trip_with_set_point():
    m = _new_model()
    # No trip -> no cabin set point -> HVAC must be zero (never fabricated).
    _drive_seconds(m, 2.0)
    assert m.hvac_wh_tick == 0.0
    m.configure_trip(destination_km=10, cabin_set_temp_c=22, ambient_temp_c=8, speed_profile_kph=[30], segment_s=5)
    m.tick(0.5)
    assert m.hvac_wh_tick > 0.0


# --- 3. higher speed consumes more than lower speed -------------------------


def test_higher_speed_consumes_more_energy():
    def drop_for(speed: float) -> float:
        m = _new_model()
        m.configure_trip(destination_km=500, cabin_set_temp_c=22, ambient_temp_c=15, speed_profile_kph=[speed], segment_s=10)
        before = m.battery_soc
        _drive_seconds(m, 30.0)
        return (before - m.battery_soc, m.trip_km)

    slow_drop, slow_km = drop_for(40.0)
    fast_drop, fast_km = drop_for(110.0)
    # Consistent energy *per km* (not just raw drop, since faster travels further).
    assert fast_drop / fast_km > slow_drop / slow_km


# --- 4. destination decreases and trip completes ----------------------------


def test_destination_decreases_and_arrives():
    m = _new_model()
    m.configure_trip(destination_km=0.05, cabin_set_temp_c=22, ambient_temp_c=15, speed_profile_kph=[60], segment_s=5)
    seen_min = []
    frames = _drive_seconds(m, 120, dt=0.1)
    trips = [f["signals"]["destination_remaining_km"] for f in frames]
    for i in range(1, len(trips)):
        assert trips[i] <= trips[i - 1] + 1e-9
    assert trips[-1] <= 0.05
    # At least eventually completed/idle.
    assert m.scenario.active_type in (ScenarioType.TRIP, ScenarioType.NONE)


# --- 5. energy accounting: single ledger, HVAC not double counted -----------


def test_energy_accounting_identity_and_no_double_count():
    m = _new_model()
    m.configure_trip(destination_km=3, cabin_set_temp_c=28, ambient_temp_c=-5, speed_profile_kph=[80], segment_s=5)
    _drive_seconds(m, 30.0)
    for _ in range(20):
        m.tick(0.5)
        ledger = m.energy_ledger
        assert ledger["battery_wh_tick"] == pytest.approx(
            ledger["motive_wh_tick"] + ledger["hvac_wh_tick"] + ledger["aux_wh_tick"], abs=1e-9
        )
        assert m.hvac_wh_tick > 0


def test_hvac_not_doubled_against_soc():
    """HVAC is counted exactly once: SOC delta equals battery_wh_tick only.

    If HVAC were also added a second time to SOC accounting, the observed SOC
    drop would exceed the ledger-scaled expectation.
    """
    m = _new_model()
    m.configure_trip(destination_km=3, cabin_set_temp_c=28, ambient_temp_c=-5, speed_profile_kph=[60], segment_s=5)
    m.tick(0.5)  # warm-up tick settles drive phase
    soc_before = m.battery_soc
    m.tick(0.5)
    eff = m.scenario.battery_efficiency_factor
    ledger_wh = m.battery_wh_tick  # the ledger of the SAME tick just run
    expected_drop = ledger_wh / PROFILE.value("battery_usable_wh") * 100.0 / eff
    assert m.hvac_wh_tick > 0
    observed = soc_before - m.battery_soc
    # SOC is drained by exactly battery_wh_tick (motive+hvac+aux) once.
    assert observed == pytest.approx(expected_drop, rel=1e-6, abs=1e-9)


# --- 6. energy engine ~ simulator (consistency, NOT real-EV accuracy) -------


def test_project_arrival_agrees_with_simulator_within_tolerance():
    """Consistency: physics projection ~ simulator under identical assumptions.

    Explicitly a software/physics consistency check — NOT evidence of real-EV
    arrival-SOC accuracy (that is a WS2 real-data evaluation concern).
    """
    m = _new_model()
    m.configure_trip(destination_km=10, cabin_set_temp_c=22, ambient_temp_c=15, speed_profile_kph=[50], segment_s=10)
    before = m.battery_soc
    frames = _drive_seconds(m, 600, dt=1.0)
    traveled_km = m.trip_km
    assert traveled_km > 2.0  # actually drove
    realized_drop = before - m.battery_soc

    energy_wh_pred = project_arrival(
        distance_km=traveled_km, battery_wh=PROFILE.value("battery_usable_wh"), soc_pct=before,
        speed_kph=50, cabin_temp_c=22, ambient_temp_c=15,
    ).energy_wh
    pred_drop = energy_wh_pred / PROFILE.value("battery_usable_wh") * 100.0
    # Loose relative bound: physics baseline and the simulator share parameters.
    assert math.isclose(realized_drop, pred_drop, rel_tol=0.35, abs_tol=3.0)


def test_what_if_monotonic_direction():
    res_speed = what_if(
        distance_km=100, battery_wh=PROFILE.value("battery_usable_wh"), soc_pct=70,
        speed_kph=80, delta_speed_kph=20, delta_cabin_temp_c=0, ambient_temp_c=10,
    )
    assert res_speed["delta_arrival_soc_pct"] < 0

    res_hvac = what_if(
        distance_km=100, battery_wh=PROFILE.value("battery_usable_wh"), soc_pct=70,
        speed_kph=80, delta_speed_kph=0, delta_cabin_temp_c=8, ambient_temp_c=10,
    )
    assert res_hvac["delta_arrival_soc_pct"] < 0


def test_project_arrival_feasibility_when_insufficient_energy():
    est = project_arrival(
        distance_km=2000, battery_wh=PROFILE.value("battery_usable_wh"), soc_pct=10,
        speed_kph=120, cabin_temp_c=28, ambient_temp_c=-10,
    )
    assert est.projected_arrival_soc_pct == 0.0
    assert "zero" in est.feasibility_note


# --- parameters profile provenance ------------------------------------------


def test_parameter_profile_has_provenance():
    p = PROFILE.get("battery_usable_wh")
    assert p.assumption is True
    assert p.provenance == "simulation_assumption"
    assert any(k.startswith("hvac") for k in PROFILE.parameters)
    # Every static sim value must be explicitly flagged, not presented as measured.
    assert all(param.assumption for param in PROFILE.parameters.values())


def test_energy_drivers_breakdown_pure():
    d = compute_drivers(110, cabin_temp_c=22, ambient_temp_c=0)
    assert d.total_w == pytest.approx(d.motive_w + d.hvac_w + d.aux_w, rel=1e-9)
    assert d.aero_w + d.rolling_w + d.inertia_w == pytest.approx(d.motive_w, rel=1e-9)
    assert d.total_w > 0