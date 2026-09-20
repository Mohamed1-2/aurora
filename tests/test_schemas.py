"""Schema validation: response contracts, severity ordering, grounding."""
import pytest
from pydantic import ValidationError

from vehicle_ai.schemas import (
    AssistantRequest,
    AssistantResponse,
    GroundingSummary,
    GatewayRequest,
    GatewayResponse,
    Message,
    SafetyVerdict,
    Severity,
    TelemetryFrame,
)


def test_severity_class():
    assert Severity.CRITICAL.value == "critical"
    assert Severity("warning") == Severity.WARNING
    assert Severity.INFO != Severity.CRITICAL
    assert issubclass(Severity, str)


def test_telemetry_frame_required_seq():
    t = TelemetryFrame(vehicle_id="v1", timestamp=1, seq=7, signals={"speed": 1.0, "distance_to_empty_km": 250.0})
    assert t.dte_km == 250.0
    with pytest.raises(ValidationError):
        TelemetryFrame(vehicle_id="v1", signals={})  # seq is required


def test_gateway_request_defaults():
    r = GatewayRequest(messages=[Message(role="user", content="hi")])
    assert r.trace_id and r.grounding is None and r.max_tokens == 600


def test_gateway_response_safety_defaults():
    g = GatewayResponse(trace_id="t", provider="mock", model="vehicle-ai-mock", content="a", latency_ms=1.0)
    assert g.refused is False and g.safety.approved is False and g.cache_hit is False
    g2 = GatewayResponse(trace_id="t", provider="mock", model="vehicle-ai-mock", content="a", latency_ms=1.0, refused=True, safety=SafetyVerdict(approved=False, blocked_reasons=["ngo"]))
    assert g2.refused is True


def test_assistant_request_validates_fields():
    with pytest.raises(ValidationError):
        AssistantRequest()  # missing question


def test_assistant_response_refused_flag():
    a = AssistantResponse(question="q", answer="a", refused=True, safety_blocked=["joke"])
    assert a.refused is True
    assert a.safety_blocked == ["joke"]
    bare = AssistantResponse(question="q", answer="a")
    assert bare.refused is False and bare.grounding.used_telemetry is False


def test_grounding_summary_defaults():
    g = GroundingSummary()
    assert g.used_telemetry is False and g.used_documents is False and g.telemetry_signals == []