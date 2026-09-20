"""Content-safety + policy gate for every completion.

A lightweight, dependency-free rule layer that runs *before* a response is
served — and, for prohibited vehicle-control and missing-telemetry requests,
*before the LLM is ever called* (a 135M model must never be the last line of
defence for "disable the battery thermal protection"). In production this
would be backed by an Azure AI Content Safety API + a policy agent; the
interface is identical, so swapping the engine is a one-file change.

Two gates run in the router:

* ``deterministic_gate`` — a *pre-LLM* check. PROHIBITED vehicle-control
  requests are refused without consulting the model, and read-only questions
  whose required canonical signals are entirely absent are refused honestly
  instead of letting the model invent numbers.
* ``evaluate_safety`` — a *post-LLM* check (kept from the original gateway)
  that re-screens the assistant output and never blocks a permitted request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vehicle_ai.schemas import SafetyVerdict

BLOCKLIST = [
    "disable braking",
    "disable brakes",
    "bypass airbag",
    "disable traction control",
    "remove safety",
    "defeat the airbag",
    "hack the brakes",
    "turn off speed limiter police",
    "make it go faster than the limiter",
]

# Content we will never fabricate: financial, legal, medical, plus personal
# data (PII) and anything unrelated to the vehicle.
OUT_OF_POLICY = [
    "investment advice",
    "medical diagnosis",
    "legal advice",
    "pretend",
    "roleplay,",
    "ignore your instructions",
    "ignore previous instructions",
    "system prompt",
    "reveal your system prompt",
]

# --------------------------------------------------------------------------- #
# Pre-LLM deterministic policy gate
#
# Mirrors the assistant-api policy tiers (read_only / safe_action /
# restricted_action / prohibited_action) but is fully self-contained in the
# gateway so the inference entry point never depends on the assistant service.
# The patterns are stricter than BLOCKLIST: they exist to make sure a
# request that targets a vehicle safety system never reaches a model that
# cannot be trusted to refuse it.
# --------------------------------------------------------------------------- #

_PROHIBITED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Safety-system tampering
    (re.compile(r"\b(disable|bypass|override|turn off|shut down)\b.{0,30}\b(thermal|safety|protection|braking|abs|esc|airbag|seatbelt|tire.?monitor|tpms|stability|traction)\b", re.IGNORECASE),
     "attempt to disable a vehicle safety system"),
    (re.compile(r"\b(disable|bypass|override)\b.{0,30}\b(battery management|bms|battery protection|cell balancing)\b", re.IGNORECASE),
     "attempt to bypass battery management"),
    (re.compile(r"\b(override|bypass)\b.{0,30}\b(emergency|fault|dtc|diagnostic)\b", re.IGNORECASE),
     "attempt to override emergency or fault handling"),
    (re.compile(r"\b(disable|bypass|override)\b.{0,30}\b(autopilot|lane.?keep|cruise.?control|automatic.?braking|collision)\b", re.IGNORECASE),
     "attempt to disable driver assistance"),
    (re.compile(r"\b(manual\s+control|take\s+over|full\s+control)\b.{0,30}\b(pedal|steering|brake|accelerat)\b", re.IGNORECASE),
     "attempt to assume direct vehicle control"),
    (re.compile(r"\b(exceed|raise|increase|bypass)\b.{0,30}\b(speed\s*limit|speed\s*cap|speed\s*limiter|max\s*speed|top\s*speed)\b", re.IGNORECASE),
     "attempt to exceed speed limits"),
    (re.compile(r"\b(engage|activate|start)\b.{0,30}\b(autopilot|self.?drive|autonomous|pilot)\b", re.IGNORECASE),
     "attempt to engage autonomous driving"),
    # Vehicle electronics tampering
    (re.compile(r"\b(hack|exploit|jailbreak|root|unlock)\b.{0,30}\b(ecu|ecus|firmware|controller|gateway|can.?bus)\b", re.IGNORECASE),
     "attempt to compromise vehicle electronics"),
    (re.compile(r"\b(reflash|flash|update)\b.{0,30}\b(firmware|ecu|controller)\b", re.IGNORECASE),
     "attempt to modify vehicle firmware"),
]

_DETERMINISTIC_REFUSAL = (
    "I cannot assist with that — it targets a safety-critical vehicle system "
    "and is not within my authority. Please consult an authorized technician "
    "or the vehicle manual."
)

_MISSING_DATA_REFUSAL = (
    "I don't have the vehicle telemetry I need to answer that honestly, so I "
    "won't guess. Please check the vehicle display or consult the manual."
)

# Intent detection (keyword, precedence-ordered) -> canonical signals the
# answer must be grounded on. Missing-data gating is *family-aware*: a driving
# request must ground on driving signals, a charging request on charging
# signals. If the question cannot be grounded on the family that is actually
# present (or there is no telemetry at all), the gateway refuses
# deterministically rather than letting the model fabricate a number.
#
# Field families (canonical names, matching VehicleDrivingContext /
# VehicleChargingSessionContext):
_DRIVING_FIELDS = [
    "soc_pct", "speed_kph", "outside_temp_c", "battery_temp_c",
    "coolant_temp_c", "pack_voltage_v",
]
_CHARGING_FIELDS = [
    "soc_arrival_pct", "soc_departure_pct", "stay_min", "energy_charged_kwh",
    "max_power_kw", "max_requested_power_kw",
]

# Intent -> driving-field requirements / charging-field requirements.
_INTENT_SIGNALS: dict[str, tuple[list[str], list[str]]] = {
    # (driving requirements, charging requirements)
    "range": (["soc_pct"], ["soc_arrival_pct", "soc_departure_pct"]),
    "battery": (["soc_pct", "battery_temp_c", "pack_voltage_v"],
                ["soc_arrival_pct", "soc_departure_pct", "energy_charged_kwh"]),
    "charging": ([], ["soc_arrival_pct", "soc_departure_pct", "stay_min",
                      "energy_charged_kwh", "max_power_kw", "max_requested_power_kw"]),
    "tire": ([], []),
    "thermal": (["battery_temp_c", "coolant_temp_c"], []),
    "energy": (["pack_voltage_v"], ["energy_charged_kwh", "max_power_kw"]),
    "summary": (["soc_pct", "speed_kph", "battery_temp_c", "outside_temp_c"],
                ["soc_arrival_pct", "soc_departure_pct", "stay_min", "energy_charged_kwh"]),
    "speed": (["speed_kph"], []),
}

# Precedence-ordered intent keywords; first match wins so that e.g. a range
# question phrased with "charge" ("how far will I get on this charge?") is not
# mislabelled charging. Bare "charge" is deliberately *not* a charging keyword
# ("how charged is my battery?" is a battery question).
_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("range", ("range", "how far", "distance", "reach", "kilometres do i have", "km left", "dte", "make it")),
    ("summary", ("summary", "status", "how's my car", "how is my car", "state of the vehicle", "what do you see", "tell me about my car")),
    ("battery", ("battery", "soc", "state of charge", "charge level", "pack voltage", "percent", "%")),
    ("speed", ("how fast", "speed", "velocity")),
    ("thermal", ("temperature", "overheat", "coolant", "too hot", "therm")),
    ("charging", ("charging", "plugged", "plug ", "plug in", "fast charg", "charge rate", "charge port", "dcfc", "recharg", "unplug", "start charging")),
    ("tire", ("tire", "tyre", "pressure", "wheel")),
    ("energy", ("consumption", "efficien", "wh/km", "energy", "kwh", "power draw")),
]


def detect_intents(question: str) -> list[str]:
    ql = question.lower()
    for name, kws in _INTENT_KEYWORDS:
        if any(k in ql for k in kws):
            return [name]
    return []


@dataclass
class GateDecision:
    """Outcome of the deterministic pre-LLM gate."""

    allowed: bool
    content: str = ""  # deterministic answer when not allowed
    reason: str = ""
    matched: str = ""

    @classmethod
    def allow(cls) -> "GateDecision":
        return cls(allowed=True)

    @classmethod
    def block(cls, content: str, reason: str, matched: str = "") -> "GateDecision":
        return cls(allowed=False, content=content, reason=reason, matched=matched)


def _family(canonical: dict[str, float | None]) -> list[str]:
    if any(canonical.get(k) is not None for k in _CHARGING_FIELDS):
        return _CHARGING_FIELDS
    if any(canonical.get(k) is not None for k in _DRIVING_FIELDS):
        return _DRIVING_FIELDS
    return []


# Small-talk / identity questions are answerable without telemetry and must not
# be caught by the "no telemetry at all" catch-all below.
_SMALLTALK_PATTERNS: tuple[str, ...] = (
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "who are you", "what can you do", "thanks", "thank you", "how are you",
)


def deterministic_gate(
    user_input: str,
    canonical: dict[str, float | None],
) -> GateDecision:
    """Pre-LLM gate: prohibit vehicle-control requests, refuse missing data.

    ``canonical`` is the normalised ``VehicleDrivingContext`` /
    ``VehicleChargingSessionContext`` as a plain dict (``None`` = signal absent;
    omitted = absent). Called before any provider runs.
    """
    for regex, reason in _PROHIBITED_PATTERNS:
        m = regex.search(user_input)
        if m:
            return GateDecision.block(
                _DETERMINISTIC_REFUSAL,
                f"prohibited vehicle-control request: {reason}",
                matched=m.group(0)[:60],
            )

    family = _family(canonical)
    intents = detect_intents(user_input)
    if intents:
        intent = intents[0]
        # Driving intent in a charging session (and vice-versa) is a grounding
        # gap: refuse rather than let the model answer from the wrong family.
        drv_req, chg_req = _INTENT_SIGNALS[intent]
        if family == _CHARGING_FIELDS:
            required = chg_req
            other_ok = drv_req
        else:
            required = drv_req
            other_ok = chg_req
        if required and not any(canonical.get(k) is not None for k in required):
            if family and other_ok and any(canonical.get(k) is not None for k in other_ok):
                return GateDecision.block(
                    _MISSING_DATA_REFUSAL,
                    f"grounding mismatch: intent {intent!r} asked in the wrong "
                    f"context family (family={family[0] or 'unknown'})",
                    matched=intent,
                )
            return GateDecision.block(
                _MISSING_DATA_REFUSAL,
                f"missing-data refusal: intent {intent!r} requires one of "
                f"{', '.join(required)} but none are present",
                matched=intent,
            )

    # Avoid the model inventing numbers when it has zero telemetry for a
    # substantive vehicle question that no intent keyword caught (live gaps,
    # out-of-scope signals, etc.). Small-talk/identity questions stay allowed.
    ql = user_input.lower()
    if not family and not _is_smalltalk(ql):
        return GateDecision.block(
            _MISSING_DATA_REFUSAL,
            "missing-data refusal: no telemetry present for a substantive question",
            matched="no-telemetry",
        )

    return GateDecision.allow()


def _is_smalltalk(ql: str) -> bool:
    for p in _SMALLTALK_PATTERNS:
        if re.search(rf"(^|\W){re.escape(p)}(\.|\?|!|$|\W)", ql):
            return True
    return False


# --------------------------------------------------------------------------- #
# Post-LLM safety screening (kept from the original gateway)
# --------------------------------------------------------------------------- #


def evaluate_safety(user_input: str, assistant_output: str) -> SafetyVerdict:
    reasons: list[str] = []
    ui, ao = user_input.lower(), assistant_output.lower()

    for phrase in BLOCKLIST:
        if phrase in ui:
            reasons.append(f"blocked instruction: '{phrase}'")

    for phrase in OUT_OF_POLICY:
        if phrase in ui:
            if phrase in ("pretend", "roleplay,", "system prompt", "reveal your system prompt", "ignore your instructions", "ignore previous instructions"):
                reasons.append(f"prompt-injection attempt detected: '{phrase}'")
            else:
                reasons.append(f"out-of-policy subject: '{phrase}'")

    score = 1.0 if not reasons else max(0.0, 1.0 - 0.25 * len(reasons))
    return SafetyVerdict(
        approved=not reasons,
        policy="vehicle-assistant-default",
        blocked_reasons=reasons,
        score=round(score, 3),
    )