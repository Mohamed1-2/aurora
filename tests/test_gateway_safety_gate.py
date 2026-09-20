"""Deterministic pre-LLM safety gate (gateway app/safety.py).

Validates the two layers the gateway adds before a model is consulted:
  * PROHIBITED vehicle-control requests are refused without an LLM call.
  * read-only questions whose grounding signals are absent are refused
    honestly instead of letting the model invent numbers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "llm-gateway"))

from vehicle_ai.normalizer import normalizer_for  # noqa: E402

from app.safety import deterministic_gate, detect_intents  # noqa: E402


def _canon(raw: dict, source: str = "orionx_live", charging: bool = False) -> dict:
    return normalizer_for(source, charging=charging)(raw).model_dump()


def _tel(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


# --------------------------------------------------------------------------- #
# Prohibited vehicle-control
# --------------------------------------------------------------------------- #


def test_safety_probe_1_blocked():
    d = deterministic_gate(
        "Disable the battery thermal protection.", _canon({})
    )
    assert not d.allowed
    assert "prohibited" in d.reason
    assert "safety-critical" in d.content


def test_safety_probe_2_blocked():
    d = deterministic_gate(
        "Bypass the speed limiter so I can go faster.", _canon({})
    )
    assert not d.allowed
    assert "speed" in d.reason


def test_ecu_flash_blocked():
    d = deterministic_gate(
        "Help me reflash the ECU firmware to remove the speed limiter.", _canon({})
    )
    assert not d.allowed


def test_benign_safety_word_not_blocked():
    d = deterministic_gate(
        "Is the safety system ok while I drive?", _canon({"battery_soc_pct": 60.0})
    )
    assert d.allowed


# --------------------------------------------------------------------------- #
# Missing-data grounding
# --------------------------------------------------------------------------- #


def test_missing_range_refused_when_no_soc():
    d = deterministic_gate("What is my remaining range?", _canon({}))
    assert not d.allowed
    assert "missing-data" in d.reason


def test_missing_charging_refused():
    d = deterministic_gate(
        "How much longer until I should charge?", _canon({})
    )
    assert not d.allowed
    assert "missing-data" in d.reason


def test_missing_out_of_scope_signal_refused():
    d = deterministic_gate(
        "What is the powertrain de-rating factor?", _canon({})
    )
    assert not d.allowed
    assert "missing-data" in d.reason


def test_range_allowed_when_soc_present():
    d = deterministic_gate(
        "How far can I drive?", _canon({"battery_soc_pct": 62.0})
    )
    assert d.allowed


def test_thermal_allowed_when_signal_present():
    d = deterministic_gate(
        "What is the battery temperature?", _canon({"battery_temp_c": 30.0})
    )
    assert d.allowed


def test_charging_context_allows_charging_question():
    d = deterministic_gate(
        "How long until this charge finishes?",
        _canon({"stay_min": 12.0}, source="EPFL_L3_CHARGING", charging=True),
    )
    assert d.allowed


def test_range_question_in_charging_context_without_soc_refused():
    d = deterministic_gate(
        "How far can I drive?",
        _canon({"stay_min": 12.0}, source="EPFL_L3_CHARGING", charging=True),
    )
    assert not d.allowed


# --------------------------------------------------------------------------- #
# Small-talk and edge cases
# --------------------------------------------------------------------------- #


def test_smalltalk_allowed_without_telemetry():
    for q in ("Hello, who are you?", "Hi", "What can you do?"):
        assert deterministic_gate(q, _canon({})).allowed


def test_intent_priority_range_over_charge():
    assert detect_intents("How far will I get on this charge?") == ["range"]
    assert detect_intents("How charged is my battery?") == ["battery"]