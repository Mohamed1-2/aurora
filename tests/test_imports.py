"""Contract, schema and import sanity for the AURORA platform."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_shared_package_imports():
    import vehicle_ai
    from vehicle_ai.schemas import (
        AssistantResponse,
        Grounding,
        GatewayRequest,
        GatewayResponse,
        Message,
        TelemetryFrame,
    )

    assert vehicle_ai.PROMPT_VERSION.startswith("aurora-prompt-v")
    assert vehicle_ai.RAG_INDEX_VERSION.startswith("rag-index-v")
    g = Grounding()
    assert g.telemetry == {} and g.documents == []
    r = GatewayResponse(trace_id="t", provider="mock", model="vehicle-ai-mock", content="ok", latency_ms=1.0)
    assert r.refused is False
    a = AssistantResponse(question="hi", answer="ok")
    assert a.refused is False
    f = TelemetryFrame(vehicle_id="ORION-X-TEST-01", seq=1, signals={"battery_soc_pct": 68.0})
    assert f.signals["battery_soc_pct"] == 68.0


def test_all_service_modules_import():
    """Every service boots its app factory without configuration errors."""
    import uvicorn

    assert hasattr(uvicorn, "run")
    services = [
        "telemetry-simulator/app/main.py",
        "llm-gateway/app/main.py",
        "rag-service/app/main.py",
        "assistant-api/app/main.py",
        "monitoring/app/main.py",
    ]
    for rel in services:
        p = ROOT / "services" / rel
        r = subprocess.run(
            [sys.executable, "-c", f"import importlib.util; s=importlib.util.spec_from_file_location('t', '{p}'); assert s and s.loader"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{rel} failed to load:\n{r.stderr}"