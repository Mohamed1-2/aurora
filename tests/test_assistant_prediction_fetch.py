"""WS3 assistant auto-fetch tests: frames -> explain-from-telemetry -> prompt block.

The assistant never computes features itself: it splits its cached frame buffer
into to/from prediction windows, posts the raw frames to the simulator's
``/api/prediction/explain-from-telemetry``, and injects the returned block into
``build_messages``. Both the split logic and the graceful fallback (no sim, too
few frames, malformed reply -> no block) are covered here.

The assistant's ``app`` package is loaded lazily inside the loader (never at
module import) so collection of OTHER test modules that own their own ``app``
(gateway, simulator) is never disturbed — the same convention the existing
``test_calibration_explain._load_assistant_py`` uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest

from vehicle_ai.schemas import AssistantRequest, Grounding, PredictionExplanation  # noqa: E402


def _load_assistant(module: str):
    """Lazy-import a module from the assistant's isolated `app` namespace."""
    for _p in (str(ROOT), str(ROOT / "services" / "assistant-api")):
        sys.path.insert(0, _p)  # assistant-api last => front of path for `app.*` resolution
    for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        sys.modules.pop(_mod, None)
    import importlib

    return importlib.import_module(module)


def _assistant_api():
    service_mod = _load_assistant("app.core.service")
    ingest_mod = _load_assistant("app.telemetry.ingest")
    prompt_mod = _load_assistant("app.core.prompt")
    config_mod = _load_assistant("app.config")
    return service_mod, ingest_mod, prompt_mod, config_mod


_PIPE = None


def _get_pipe():
    """Single AssistantPipeline across tests: constructing it registers the
    same prometheus metric names twice -> DuplicateTimeseries otherwise."""
    global _PIPE
    service_mod, ingest_mod, *_ = _assistant_api()
    if _PIPE is None:
        _PIPE = service_mod.AssistantPipeline(ingest_mod.TelemetryIngest(ws_url="ws://x/ws", replay_path=""))
    return _PIPE


@pytest.fixture()
def pipeline():
    _, ingest_mod, _, _ = _assistant_api()
    telemetry = ingest_mod.TelemetryIngest(ws_url="ws://x/ws", replay_path="", window_seconds=10.0, max_frames=10000)
    pipe = _get_pipe()
    pipe.telemetry = telemetry  # per-test fresh frame buffer
    yield pipe, telemetry


def _make_frames(n: int) -> list[dict]:
    return [
        {"vehicle_id": "ORION-X-TEST-01", "seq": i, "signals": {"vehicle_speed_kph": 80.0}}
        for i in range(n)
    ]


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _explanation_payload():
    return {
        "shape": "absolute",
        "status": "active",
        "model_version": "v2-ws1-r1",
        "capacity_kwh": 58.0,
        "to_prediction": {
            "physics_soc_drop_pct": 1.45, "ml_residual_pct": 0.19,
            "calibrated_soc_drop_pct": 1.64, "speed_band": "highway",
            "lambda_applied": 0.8, "model_version": "v2-ws1-r1", "status": "active",
        },
        "from_prediction": None,
        "physics_contributions": [],
        "ml_contributions": [],
        "top_causes": [{"key": "rolling", "label": "Rolling resistance", "contribution_soc_pct": 0.77, "provenance": "physics", "direction": "increases_drop"}],
        "identity_checks": {"physics_sum_matches": True, "ml_sum_matches": True, "blend_matches": True, "delta_matches": None, "tolerance_soc_pct": 1e-6},
    }


def test_split_single_window_and_post_frames(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    telemetry.frames.extend(_make_frames(2000))

    seen: dict = {}

    class _FakeClient:
        async def post(self, url, json):
            seen["url"] = url
            seen["json"] = json
            return _FakeResp(_explanation_payload())

    pipe._client = _FakeClient()  # type: ignore[assignment]

    expl = asyncio.run(pipe._fetch_prediction_explanation())
    assert expl is not None and expl.status == "active"
    assert seen["url"].endswith("/api/prediction/explain-from-telemetry")
    body = seen["json"]
    assert len(body["to_frames"]) == 2000  # fewer frames than the full window -> all of them
    assert body["from_frames"] is None  # buffer not long enough for a delta
    assert body["hz"] == 10.0


def test_split_uses_delta_when_buffer_holds_two_windows(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    telemetry.frames.extend(_make_frames(2 * 3000 + 100))  # > 2 full 300 s windows

    seen: dict = {}

    class _FakeClient:
        async def post(self, url, json):
            seen["json"] = json
            return _FakeResp(_explanation_payload())

    pipe._client = _FakeClient()  # type: ignore[assignment]

    _, _, _, config_mod = _assistant_api()
    window_frames = int(config_mod.settings.prediction_window_seconds * 10)
    expl = asyncio.run(pipe._fetch_prediction_explanation())
    assert expl is not None
    assert len(seen["json"]["to_frames"]) == window_frames
    assert len(seen["json"]["from_frames"]) == window_frames


def test_fetch_returns_none_when_too_few_frames(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    telemetry.frames.extend(_make_frames(100))  # < prediction_min_frames

    called = []

    class _FakeClient:
        async def post(self, url, json):  # pragma: no cover - must not be called
            called.append(True)
            return _FakeResp(_explanation_payload())

    pipe._client = _FakeClient()  # type: ignore[assignment]
    assert asyncio.run(pipe._fetch_prediction_explanation()) is None
    assert called == []


def test_fetch_graceful_on_transport_error(pipeline):
    import asyncio

    import httpx

    pipe, telemetry = pipeline
    telemetry.frames.extend(_make_frames(1000))

    async def boom(self, url, json):  # noqa: ANN001
        raise httpx.ConnectError("sim unreachable")

    pipe._client = type("C", (), {"post": boom})()  # type: ignore[assignment]
    assert asyncio.run(pipe._fetch_prediction_explanation()) is None


def test_fetch_graceful_on_malformed_reply(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    telemetry.frames.extend(_make_frames(1000))

    class _FakeClient:
        async def post(self, url, json):  # noqa: ANN001
            return _FakeResp({"status": "active"})  # missing model version/fields

    pipe._client = _FakeClient()  # type: ignore[assignment]
    assert asyncio.run(pipe._fetch_prediction_explanation()) is None


def test_block_injected_when_fetched(pipeline):
    """The ask() path puts a fetched explanation into build_messages."""
    _, _, prompt_mod, _ = _assistant_api()
    expl = PredictionExplanation.model_validate(_explanation_payload())
    msgs = prompt_mod.build_messages("Why did my SOC drop?", Grounding(), intents=["why_prediction"], explanation=expl)
    assert "## PREDICTION EXPLANATION" in msgs[1].content
    assert "1.64 SOC pts" in msgs[1].content

    # intent absent => fetched explanation must NOT inject a block
    msgs2 = prompt_mod.build_messages("How far can I go?", Grounding(), intents=["range"], explanation=expl)
    assert "## PREDICTION EXPLANATION" not in msgs2[1].content


# ---------------------------------------------------------------------------
# Deterministic driving-time path (no LLM traffic, no service calls)
# ---------------------------------------------------------------------------


def _driving_time_frames(n: int) -> list[dict]:
    return [
        {
            "vehicle_id": "ORION-X-TEST-01",
            "seq": i,
            "timestamp": "2026-01-01T00:00:00Z",
            "signals": {"vehicle_speed_kph": 80.0, "distance_to_empty_km": 100.0},
        }
        for i in range(n)
    ]


def test_ask_uses_deterministic_driving_time_path(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    telemetry.frames.extend(_driving_time_frames(200))
    resp = asyncio.run(pipe.ask(AssistantRequest(question="how long can I drive?", ground_telemetry=True)))

    assert resp.provider == "deterministic"
    assert resp.model == "driving-time-estimator"
    assert resp.refused is False
    assert "driving_time" in resp.intents
    assert resp.grounding.used_telemetry is True
    assert resp.citations == []
    assert "1 h 15 min" in resp.answer  # 100 km / 80 km/h * 60


def test_ask_refuses_when_no_telemetry(pipeline):
    import asyncio

    pipe, telemetry = pipeline
    resp = asyncio.run(pipe.ask(AssistantRequest(question="how much driving time do I have left?", ground_telemetry=True)))

    assert resp.refused is True
    assert resp.model == "driving-time-refusal"
    assert resp.grounding.used_telemetry is False
    assert "can't give a reliable" in resp.answer