"""WS6 Scenario Director tests.

The cockpit's one-click scenarios must operate the *real* vehicle model: each
step is a genuine simulator command and nothing is faked in the assistant. These
tests pin:

* ``charge_stall``: vehicle parks and charges, but ``charge_rate_kw`` stays 0
  (the proactive ``charging_stall`` detector's precondition);
* ``low_energy``: the pack is clamped to the ~14.5% floor, i.e. reachable
  "low arrival energy" without waiting an hour of simulated driving;
* outside temperature now tracks the trip's ``ambient_temp_c`` (HIGHWAY+COLD);
* the assistant ``/scenarios`` orchestrator relays exact step sequences and
  reports the simulator's own control state; an in-sequence failure aborts
  honestly (``complete=false``) instead of faking success;
* ``/explain/why`` forwards the live window to the deterministic explainer and
  styles the ranked causes; ``/explain/whatif`` builds a real physics recalc
  with deltas computed from the live frame.

The intelligence stack (calibrated prediction, proactive engine, RAG, gateway)
remains frozen; these routes only relay to it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent

# --- simulator vehicle model (evict foreign `app` packages first) -----------
for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_mod, None)
sys.path.insert(0, str(ROOT / "services" / "telemetry-simulator"))
sys.path.insert(0, str(ROOT))

from app.engine.scenarios import ScenarioType  # noqa: E402
from app.engine.vehicle_model import VehicleModel  # noqa: E402

# evict again: this file sorts alphabetically before the llm-gateway tests, so
# leaving the telemetry-sim `app` cached would shadow `app.providers`/`app.safety`.
for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_mod, None)


def _new_model(seed: int = 7) -> VehicleModel:
    return VehicleModel("T1", seed=seed)


# --- simulator scenario semantics -------------------------------------------


def test_charge_stall_powers_at_zero_while_charging():
    m = _new_model()
    m.configure(ScenarioType.CHARGE_STALL, 300.0, 1.0)
    frame = m.tick(0.5)  # park-and-charge enters on the first tick
    assert frame["signals"]["charge_port_latched"] == 1.0
    assert frame["signals"]["charge_rate_kw"] == 0.0

    status = m.control_status()
    assert status["charging"] is True
    assert status["charge_rate_kw"] == 0.0
    assert status["scenario"] == "charge_stall"

    before = m.battery_soc
    for _ in range(120):  # 60 s of "charging"
        m.tick(0.5)
    assert m.battery_soc == pytest.approx(before)  # no energy arrived


def test_regular_charge_still_pushes_power():
    m = _new_model()
    m.configure(ScenarioType.CHARGE, 60.0, 1.0)
    before = m.battery_soc
    frame = m.tick(5.0)
    assert frame["signals"]["charge_rate_kw"] > 0
    assert m.battery_soc > before  # nominal charge actually raises SOC


def test_low_energy_clamps_pack_to_floor():
    m = _new_model()
    m.configure(ScenarioType.LOW_ENERGY, 60.0, 1.0)
    assert m.battery_soc <= 14.5
    frame = m.tick(0.5)
    assert frame["signals"]["battery_soc_pct"] <= 14.5
    assert frame["signals"]["distance_to_empty_km"] < 60  # long-range-risk regime
    assert m.control_status()["scenario"] == "low_energy"


def test_string_scenario_change_of_charge_rearms_driving():
    m = _new_model()
    m.configure(ScenarioType.CHARGE_STALL, 300.0, 1.0)
    m.tick(0.5)
    assert m.charging is True
    m.configure(ScenarioType.BATTERY_HEAT, 360.0, 1.0)
    assert m.charging is False          # leaving the charging yard re-arms driving
    assert m.control_status()["scenario"] == "battery_heat"
    for _ in range(20):                  # 10 s
        m.tick(0.5)
    assert m.speed > 5                   # moving again, not parked on a stale timer


def test_outside_temperature_tracks_trip_ambient():
    m = _new_model()
    m.configure_trip(destination_km=80, cabin_set_temp_c=24, ambient_temp_c=-5.0,
                     speed_profile_kph=[110], segment_s=5)
    frame = m.tick(0.5)
    assert frame["signals"]["outside_temp_c"] == -5.0
    assert frame["signals"]["ambient_temp_c"] == -5.0
    assert m.control_status()["ambient_temp_c"] == -5.0

    free = _new_model()  # free drive keeps the nominal outdoor band
    frame = free.tick(0.5)
    assert frame["signals"]["outside_temp_c"] is not None
    assert 14.0 <= frame["signals"]["outside_temp_c"] <= 22.0


def test_control_status_exposes_scenario_and_trip():
    m = _new_model()
    m.configure(ScenarioType.BATTERY_HEAT, 360.0, 1.0)
    status = m.control_status()
    assert status["scenario"] == "battery_heat"
    assert status["trip_active"] is False
    assert status["destination_km"] is None
    assert status["remaining_km"] is None

    m.configure_trip(destination_km=55, cabin_set_temp_c=20, ambient_temp_c=4.0,
                     speed_profile_kph=[100], segment_s=5)
    status = m.control_status()
    assert status["trip_active"] is True
    assert status["destination_km"] == pytest.approx(55.0)
    assert status["cabin_set_temp_c"] == pytest.approx(20.0)


# --- assistant routes (director / why / what-if) -----------------------------


def _load_routes():
    """Load the assistant's ``app/api/routes.py`` by file (``app`` collision-safe)."""
    path = ROOT / "services" / "assistant-api" / "app" / "api" / "routes.py"
    sys.path.insert(0, str(ROOT / "services" / "assistant-api"))
    spec = importlib.util.spec_from_file_location("aurora_test_routes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aurora_test_routes"] = mod
    spec.loader.exec_module(mod)
    return mod


class RecordingClient:
    """Scripted vehicle client that records every relayed call."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, object]] = []

    async def request(self, method: str, url: str, **kwargs) -> SimpleNamespace:
        self.calls.append((method.upper(), url, kwargs.get("json")))
        payload = self.responses.pop(0)
        return SimpleNamespace(ok=True, raise_for_status=lambda: None, json=lambda: payload)


def _state(client: RecordingClient, extra: dict | None = None) -> SimpleNamespace:
    state = SimpleNamespace(
        settings=SimpleNamespace(telemetry_http="http://vehicle.test"),
        pipeline=SimpleNamespace(_client=client),
    )
    if extra:
        for key, value in extra.items():
            setattr(state, key, value)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_scenario_catalog_lists_all_director_keys():
    routes = _load_routes()
    assert [s["key"] for s in routes.scenarios()] == [
        "normal", "highway_cold", "low_arrival", "tire_leak", "charging_stall", "battery_heat",
    ]


def test_orchestrator_relays_exact_step_sequence_highway_cold():
    routes = _load_routes()
    responses = [
        {"ack": True},
        {"started": True},   # trip
        {"ack": True},       # speed override
        {"trip_active": True, "scenario": "trip", "ambient_temp_c": -5.0},
    ]
    resp = asyncio.run(routes.launch_scenario("highway_cold", _state(RecordingClient(responses))))

    assert resp["complete"] is True
    assert [s["method"] for s in resp["steps"]] == ["POST", "POST", "POST"]
    assert [s["path"] for s in resp["steps"]] == ["/api/scenario", "/api/trip", "/api/control/speed"]
    assert resp["steps"][0]["body"] == {"type": "none", "duration_s": 1, "intensity": 0}
    assert resp["steps"][1]["body"]["ambient_temp_c"] == -5.0
    assert resp["steps"][2]["body"] == {"speed_kph": 120}
    assert resp["control"]["ambient_temp_c"] == -5.0


def test_whatif_body_and_battery_units_match_sim():
    routes = _load_routes()
    responses = [
        {"cabin_set_temp_c": 22.0, "ambient_temp_c": -5.0, "trip_active": True, "remaining_km": 48.0},
        {"baseline_arrival_soc_pct": 21.0, "perturbed_arrival_soc_pct": 24.6, "delta_arrival_soc_pct": 3.6},
    ]
    state = SimpleNamespace(
        telemetry=SimpleNamespace(frames=[
            {"signals": {"battery_soc_pct": 60.0, "vehicle_speed_kph": 101.0, "outside_temp_c": -5.0}},
        ]),
    )
    req = _state(RecordingClient(responses), extra={"telemetry": state.telemetry})
    out = asyncio.run(routes.explain_whatif(routes.WhatIfRequest(speed_kph=90, cabin_temp_c=20), req))

    body = out["request"]
    assert body["battery_wh"] == 82_000.0
    assert body["delta_speed_kph"] == pytest.approx(-11.0)
    assert body["delta_cabin_temp_c"] == pytest.approx(-2.0)
    assert body["distance_km"] == pytest.approx(48.0)
    assert "+3.6 points" in out["narrative"]


def test_why_forwards_live_window_and_styles_ranking():
    routes = _load_routes()

    class FakeExplanation:
        def model_dump(self, mode="json"):
            return {
                "status": "active",
                "to_prediction": {"calibrated_soc_drop_pct": 2.71, "status": "active"},
                "top_causes": [
                    {"key": "speed_kph", "label": "speed", "contribution_soc_pct": 1.9, "direction": "higher speed"},
                    {"key": "hvac_power_kw", "contribution_soc_pct": -0.4, "direction": "climate"},
                ],
            }

    async def fetch():
        return FakeExplanation()

    pipeline = SimpleNamespace(_fetch_prediction_explanation=fetch)
    state = SimpleNamespace(pipeline=pipeline)
    req = SimpleNamespace(app=SimpleNamespace(state=state))
    out = asyncio.run(routes.explain_why(req))

    assert out["kind"] == "why"
    assert "largest driver is speed" in out["narrative"]
    assert "1.90 points" in out["narrative"]

    async def none_fetch():
        return None

    pipeline._fetch_prediction_explanation = none_fetch
    out = asyncio.run(routes.explain_why(req))
    assert out["status"] == "collecting"
    assert "keep driving" in out["narrative"]


def test_unknown_scenario_is_404():
    from fastapi import HTTPException

    routes = _load_routes()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.launch_scenario("teleport", _state(RecordingClient([{}]))))
    assert exc_info.value.status_code == 404


def test_orchestrator_aborts_honestly_on_in_sequence_failure():
    routes = _load_routes()

    class FlakyClient:
        def __init__(self):
            self.calls = []

        async def request(self, method, url, **kwargs):
            self.calls.append(url)
            if url.endswith("/api/trip"):
                raise httpx.ConnectError("boom", request=httpx.Request(method, url))
            return SimpleNamespace(ok=True, raise_for_status=lambda: None, json=lambda: {"ack": True})

    req = _state(FlakyClient())
    resp = asyncio.run(routes.launch_scenario("highway_cold", req))
    assert resp["complete"] is False
    assert "vehicle driver unavailable" in resp["error"]
    assert resp["steps"][0]["path"] == "/api/scenario"  # prior step recorded
    assert resp["steps"][0]["body"] == {"type": "none", "duration_s": 1, "intensity": 0}