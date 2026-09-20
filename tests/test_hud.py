"""WS5 HUD facade tests — thin aggregation, provenance-first labelling.

The facade adds no inference: it labels values their owners already decided.
These tests pin:

* the provenance taxonomy (measured / prediction / derived / documentation /
  simulator) is present and each signal carries one;
* absent signals are reported ``present=false`` with ``value=None`` (silence,
  never a fabricated zero);
* simulated vs measured selection follows the environment contract;
* coverage math and status transitions for the prediction block;
* blocks degrade to explicit ``error``/``unreachable`` labels instead of
  invented numbers.

The assistant's ``app`` package is loaded lazily (subprocess-safe, matching
``test_calibration_explain``), so gateway/simulator's own ``app`` modules are
never disturbed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest  # noqa: E402


def _load_hud():
    """Load the assistant's ``app/api/hud.py`` by file, not by package name.

    Other services ship their own ``app`` package (gateway, simulator) and may
    already occupy ``sys.modules["app"]`` when this module loads — importing
    ``from app.api import hud`` would then resolve to the wrong service. Loading
    the file directly avoids the collision entirely.
    """
    path = ROOT / "services" / "assistant-api" / "app" / "api" / "hud.py"
    spec = importlib.util.spec_from_file_location("aurora_test_hud", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aurora_test_hud"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def _hud():
    return _load_hud()


def _fake_telemetry():
    class _T:
        window_seconds = 10.0

        @property
        def frames(self):
            return [_F()] * 600

        def grounding(self):
            return {
                "vehicle_id": "ORION-X-TEST-01",
                "mode": "normal",
                "charging": False,
                "source": "live",
                "signals": {"battery_soc_pct": 61.4, "tire_pressure_rr_kpa": 221.1},
                "mean": {"battery_soc_pct": 61.0},
                "min": {"battery_soc_pct": 60.9},
                "max": {"battery_soc_pct": 62.0},
                "trending": {"battery_soc_pct": -0.12},
                "dtcs": [],
                "warnings": [],
            }

        def signal_series(self, signal, max_age_s=60.0, stride_s=1.0):
            return [[1810000000.0, 62.0], [1810000010.0, 61.4]] if signal == "battery_soc_pct" else []

    class _F:
        vehicle_id = "ORION-X-TEST-01"
        signals = {}

    return _T()


class FakeSnap:
    generated_at = 1810000000.0
    poll_interval_s = 5.0
    detectors_eligible = ["low_range"]
    detectors_skipped_no_signal = {"prediction_discharge_risk": ["<300 s coverage: 600 frames>"]}
    active = []
    recent = []


def test_provenance_legend_covers_all_five(_hud):
    assert set(_hud.PROVENANCE_LEGEND) == {"measured", "prediction", "derived", "documentation", "simulator"}


def test_signals_metadata_labels_provenance_and_absences(_hud):
    _t = _fake_telemetry()
    meta = _hud.signals_metadata(_t.grounding(), simulated=True, spark_getter=lambda s: [[1.0, 2.0]])
    soc = meta["battery_soc_pct"]
    assert soc["provenance"] == "simulator"
    assert soc["present"] is True
    assert soc["value"] == 61.4
    assert soc["unit"] == "%"
    assert soc["window_min"] == 60.9
    assert soc["trending_per_min"] == -0.12
    assert soc["spark"] == [[1.0, 2.0]]

    missing = meta["battery_voltage_v"]
    assert missing["present"] is False
    assert missing["value"] is None
    assert missing["spark"] is None  # absence is honest, never zero


def test_simulated_vs_measured_provenance(_hud):
    assert _hud.signals_metadata(_fake_telemetry().grounding(), simulated=False, spark_getter=lambda s: [])["battery_soc_pct"]["provenance"] == "measured"


def test_telemetry_block_no_grounding_is_honest(_hud):
    class _Empty:
        window_seconds = 10.0
        frames = []
        grounding = lambda self: {}
        signal_series = lambda self, s, **k: []

    b = _hud.telemetry_block(_Empty(), simulated=True)
    assert b["status"] == "no-telemetry"
    assert b["signals"] == {}
    assert b["provenance"] == "simulator"


def test_prediction_block_coverage_and_status(_hud):
    low = _hud.prediction_block(frame_count=148, service_status={"active": True}, explanation=None, min_frames=300)
    assert low["status"] == "collecting"
    assert low["coverage_pct"] < 10.0
    assert low["provenance"] == "prediction"

    active = _hud.prediction_block(frame_count=1240, service_status={"active": True}, explanation=None, min_frames=300)
    assert active["status"] == "active"
    assert active["coverage_pct"] == pytest.approx(41.3, abs=0.1)

    fallback = _hud.prediction_block(frame_count=1240, service_status={"active": False}, explanation=None, min_frames=300)
    assert fallback["status"] == "physics_only_fallback"

    with_expl = _hud.prediction_block(frame_count=1240, service_status={"active": True}, explanation={"status": "active"}, min_frames=300)
    assert with_expl["status"] == "active"
    assert with_expl["explanation"]["status"] == "active"


def test_models_and_documents_degrade_explicitly(_hud):
    m = _hud.models_block(None, None, None)
    assert "error" in m["providers"]
    assert m["provenance"] == "derived"
    d = _hud.documents_block(None)
    assert d["index"]["error"]
    assert d["provenance"] == "documentation"
    assert "not the manufacturer" in d["label"]


def test_assemble_contract_shape(_hud):
    now = 1810000000.0
    identity = _hud.identity_block("local", True)
    telemetry = {"vehicle_id": "ORION-X-TEST-01", "mode": "normal", "charging": False, "provenance": "simulator"}
    payload = _hud.assemble(
        now=now,
        identity=identity,
        telemetry=telemetry,
        proactive=None,
        prediction=_hud.prediction_block(0, None, None, 300),
        models=_hud.models_block({"mock": {"configured": True}}, None, ["mock"]),
        documents=_hud.documents_block({"n_chunks": 403}),
        services=_hud.services_block("ok", "ok", "ok"),
    )
    assert payload["v"] == "v1"
    assert payload["generated_at"] == now
    assert payload["vehicle"]["id"] == "ORION-X-TEST-01"
    assert payload["proactive"] is None
    assert set(payload["provenance_legend"]) == {"measured", "prediction", "derived", "documentation", "simulator"}
    assert payload["identity"]["telemetry_provenance"] == "simulator"
    assert payload["identity"]["run"] == "simulated eval stack"