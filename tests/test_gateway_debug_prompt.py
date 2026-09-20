"""Gateway /v1/debug/prompt + provider-pinned cache-key tests (no torch needed)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "llm-gateway"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


class TestDebugPromptEndpoint:
    def test_requires_provider(self):
        r = client.post(
            "/v1/debug/prompt",
            json={"messages": [{"role": "user", "content": "hi"}], "provider": None},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["error"] == "provider required"
        assert body["prompt"] is None

    def test_unknown_provider(self):
        r = client.post(
            "/v1/debug/prompt",
            json={"messages": [{"role": "user", "content": "hi"}], "provider": "does-not-exist"},
        )
        assert r.json()["error"] == "unknown provider 'does-not-exist'"

    def test_mock_has_no_renderer(self):
        r = client.post(
            "/v1/debug/prompt",
            json={"messages": [{"role": "user", "content": "hi"}], "provider": "mock"},
        )
        assert r.json()["prompt"] is None

    def test_bad_request_still_schema_validated(self):
        r = client.post("/v1/debug/prompt", json={})
        assert r.status_code == 422