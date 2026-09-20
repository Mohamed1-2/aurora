"""Live driver command surface tests.

The immersive cockpit (WS6) lets the driver command speed / climate /
destination / scenarios through the assistant relay. These tests pin that:

* ``set_target_speed`` actually moves the vehicle toward the commanded cruise,
  and releasing it resumes the active plan (trip profile);
* ``set_cabin_set_temp`` only applies inside an active trip — WS1 never
  fabricates an HVAC set point for non-trip telemetry;
* the assistant's ``_relay_sim`` forwards driver commands to the vehicle and
  surfaces a rejecting/unreachable vehicle as an explicit HTTP error, never a
  fake acknowledgement.

The intelligence stack (proactive engine, calibrated prediction, RAG, LLM
gateway) is untouched by design — the commands only operate the vehicle model
under test, feeding the same frames those frozen layers reason over.
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

# evict the telemetry-sim `app` package again: this file sorts before the
# llm-gateway tests at collection time, so leaving it cached would shadow the
# gateway's `app.providers`/`app.safety` during their import.
for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_mod, None)


def _new_model(seed: int = 20240627) -> VehicleModel:
    return VehicleModel("T1", seed=seed)


def _drive_seconds(model: VehicleModel, seconds: float, dt: float = 0.5) -> list[dict]:
    return [model.tick(dt) for _ in range(int(seconds / dt))]


def test_cruise_override_moves_vehicle_and_release_resumes_plan():
    m = _new_model()
    m.configure_trip(destination_km=50, cabin_set_temp_c=22, ambient_temp_c=15,
                     speed_profile_kph=[40], segment_s=5)
    _drive_seconds(m, 10.0)  # settle onto the 40 kph profile
    assert abs(m.speed - 40) < 4.0

    m.set_target_speed(100)
    assert m.control_status()["override_speed_kph"] == 100.0
    _drive_seconds(m, 12.0)
    assert m.speed > 70  # pulled up toward the commanded cruise
    top = m.control_status()
    assert top["target_speed_kph"] == pytest.approx(100.0, abs=1.5)
    assert top["mode"] == "scenario"

    m.set_target_speed(None)
    assert m.control_status()["override_speed_kph"] is None
    _drive_seconds(m, 30.0)
    assert abs(m.speed - 40) < 1.5  # released override -> follows the profile again


def test_cabin_set_temp_applies_only_in_trip():
    m = _new_model()
    m.configure_trip(destination_km=20, cabin_set_temp_c=21, ambient_temp_c=10,
                     speed_profile_kph=[60], segment_s=5)
    assert m.set_cabin_set_temp(28) == 28.0
    frame = m.tick(0.5)
    assert frame["signals"]["cabin_set_temp_c"] == 28.0
    assert frame["signals"]["hvac_power_kw"] > 0
    status = m.control_status()
    assert status["cabin_set_temp_c"] == 28.0
    assert status["hvac_power_kw"] > 0

    parked = _new_model()  # free drive, no trip -> climate refuses honestly
    assert parked.set_cabin_set_temp(24) is None
    assert parked.control_status()["cabin_set_temp_c"] is None
    assert parked.control_status()["hvac_power_kw"] == 0.0


def test_control_status_shape_and_trip_clears_override():
    m = _new_model()
    m.set_target_speed(88.0)
    before = m.control_status()
    assert before["override_speed_kph"] == 88.0
    assert before["target_speed_kph"] == 88.0
    m.configure_trip(destination_km=5, cabin_set_temp_c=22, ambient_temp_c=15,
                     speed_profile_kph=[55], segment_s=5)
    assert m.control_status()["override_speed_kph"] is None
    m.tick(0.5)
    assert m.scenario.active_type == ScenarioType.TRIP
    assert m.control_status()["target_speed_kph"] == pytest.approx(55.0, abs=1.0)


# --- assistant relay --------------------------------------------------------


def _load_routes():
    """Load the assistant's ``app/api/routes.py`` by file (``app`` collision-safe)."""
    path = ROOT / "services" / "assistant-api" / "app" / "api" / "routes.py"
    sys.path.insert(0, str(ROOT / "services" / "assistant-api"))
    spec = importlib.util.spec_from_file_location("aurora_test_routes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aurora_test_routes"] = mod
    spec.loader.exec_module(mod)
    return mod


def _relay_request():
    class FakeClient:
        async def request(self, method, url, **kwargs):
            return SimpleNamespace(ok=True, raise_for_status=lambda: None, json=lambda: {"ack": True})

    state = SimpleNamespace(
        settings=SimpleNamespace(telemetry_http="http://vehicle.test"),
        pipeline=SimpleNamespace(_client=FakeClient()),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_relay_forwards_driver_command_and_returns_payload():
    routes = _load_routes()
    req = _relay_request()
    payload = asyncio.run(routes._relay_sim(req, "POST", "/api/control/speed", json={"speed_kph": 70}))
    assert payload == {"ack": True}


def test_relay_surfaces_unreachable_vehicle_as_explicit_error():
    from fastapi import HTTPException

    routes = _load_routes()

    class BoomClient:
        async def request(self, method, url, **kwargs):
            raise httpx.ConnectError("refused", request=httpx.Request(method, url))

    state = SimpleNamespace(
        settings=SimpleNamespace(telemetry_http="http://vehicle.test"),
        pipeline=SimpleNamespace(_client=BoomClient()),
    )
    req = SimpleNamespace(app=SimpleNamespace(state=state))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes._relay_sim(req, "GET", "/api/control"))
    assert exc_info.value.status_code == 502  # vehicle unreachable -> never a fake ack
    assert "vehicle driver unavailable" in exc_info.value.detail