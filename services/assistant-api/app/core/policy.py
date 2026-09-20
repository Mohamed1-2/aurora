"""Safety policy layer for the in-vehicle AI assistant.

Classifies every incoming request into one of four tiers before it reaches the LLM:

  READ_ONLY        informational queries answered from telemetry + KB (range, battery, etc.)
  SAFE_ACTION       actions the vehicle can safely perform autonomously
  RESTRICTED_ACTION actions requiring explicit driver confirmation (future: "start charging")
  PROHIBITED_ACTION must NEVER be delegated to an LLM — deterministic refusal only

This is the architectural boundary that separates a vehicle AI assistant from a
generic enterprise chatbot.  The LLM never sees PROHIBITED requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from vehicle_ai.logging_conf import get_logger

logger = get_logger("aurora.assistant.policy")


class PolicyLevel(str, Enum):
    READ_ONLY = "read_only"
    SAFE_ACTION = "safe_action"
    RESTRICTED_ACTION = "restricted_action"
    PROHIBITED_ACTION = "prohibited_action"


@dataclass
class PolicyDecision:
    level: PolicyLevel
    reason: str = ""
    matched_pattern: str = ""
    allowed_to_llm: bool = True
    deterministic_answer: str = ""

    def __post_init__(self) -> None:
        self.allowed_to_llm = self.level in (PolicyLevel.READ_ONLY, PolicyLevel.SAFE_ACTION)


# ---------------------------------------------------------------------------
# Pattern definitions
#
# Each pattern is (compiled regex, PolicyLevel, human reason).
# Patterns are evaluated first-match-wins in order.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], PolicyLevel, str]] = []


def _pat(pattern: str, level: PolicyLevel, reason: str) -> None:
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), level, reason))


# --- PROHIBITED: must never reach the LLM ---

# Safety system tampering
_pat(r"\b(disable|bypass|override|turn off|shut down)\b.{0,30}\b(thermal|safety|protection|braking|abs|esc|airbag|seatbelt|tire.?monitor|tpms| stability|traction)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to disable a vehicle safety system")
_pat(r"\b(disable|bypass|override)\b.{0,30}\b(battery management|bms|battery protection|cell balancing)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to bypass battery management")
_pat(r"\b(override|bypass)\b.{0,30}\b(emergency|fault|dtc|diagnostic)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to override emergency or fault handling")
_pat(r"\b(overwrite|delete|clear)\b.{0,30}\b(fault|dtc|diagnostic|error)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to tamper with diagnostic records")

# Safety-critical control
_pat(r"\b(disable|bypass|override)\b.{0,30}\b(autopilot|lane.?keep|cruise.?control|automatic.?braking|collision)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to disable driver assistance")
_pat(r"\b(manual\s+control|take\s+over|full\s+control)\b.{0,30}\b(pedal|steering|brake|accelerat)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to assume direct vehicle control")
_pat(r"\b(engage|activate|start)\b.{0,30}\b(autopilot|self.?drive|autonomous|pilot)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to engage autonomous driving")
_pat(r"\b(exceed|raise|increase|bypass)\b.{0,30}\b(speed\s*limit|speed\s*cap|max\s*speed|top\s*speed)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to exceed speed limits")

# Privacy / data
_pat(r"\b(upload|send|transmit|share)\b.{0,30}\b(location|gps|position|driving\s+data|telemetry|personal)\b.{0,20}\b(third.party|external|server|cloud|api)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to exfiltrate personal/driving data")
_pat(r"\b(hack|exploit|jailbreak|root|unlock)\b.{0,30}\b(ecu|ecus|firmware|controller|gateway|can.?bus)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to compromise vehicle electronics")
_pat(r"\b(reflash|flash|update)\b.{0,30}\b(firmware|ecu|controller)\b",
     PolicyLevel.PROHIBITED_ACTION, "Attempt to modify vehicle firmware")

# --- RESTRICTED: allowed but must confirm (future) ---

_pat(r"\b(start|begin)\b.{0,30}\b(charging|dcfc|fast.?charg)\b",
     PolicyLevel.RESTRICTED_ACTION, "Initiate charging requires driver confirmation")
_pat(r"\b(stop|cancel|halt)\b.{0,30}\b(charging)\b",
     PolicyLevel.RESTRICTED_ACTION, "Stop charging requires driver confirmation")
_pat(r"\b(set|change|adjust)\b.{0,30}\b(climate|temperature|hvac|heat|air.?condition)\b",
     PolicyLevel.RESTRICTED_ACTION, "Climate control change requires driver confirmation")
_pat(r"\b(unlock|lock|open|close)\b.{0,30}\b(doors?|trunk|boot|hood|frunk|charge.?port)\b",
     PolicyLevel.RESTRICTED_ACTION, "Vehicle access control requires driver confirmation")
_pat(r"\b(precondition|pre.?condition|preheat|pre.?heat)\b.{0,30}\b(battery|pack|cabin)\b",
     PolicyLevel.RESTRICTED_ACTION, "Preconditioning requires driver confirmation")
_pat(r"\b(remind|notification|alert)\b.{0,30}\b(me|driver|owner)\b",
     PolicyLevel.RESTRICTED_ACTION, "Driver notification requires confirmation")

# --- SAFE_ACTION: informational with side-effects ---

_pat(r"\b(send|email|share)\b.{0,30}\b(report|summary|diagnostic)\b",
     PolicyLevel.SAFE_ACTION, "Share/report requests are informational")
_pat(r"\b(export|download|save)\b.{0,30}\b(log|report|summary|data)\b",
     PolicyLevel.SAFE_ACTION, "Export requests are informational")

# --- READ_ONLY: everything else defaults here via the catch-all below ---

# Explicit READ_ONLY patterns (for logging / future filtering)
_pat(r"\b(what|how|show|tell|check|describe|explain|status|summary|health|level|remaining|range|pressure|temperature|power|consumption|maintenance|service|fault|code|dtc)\b",
     PolicyLevel.READ_ONLY, "Informational query")


def classify(question: str, intents: list[str]) -> PolicyDecision:
    """Classify a user question into a policy tier.

    Returns a ``PolicyDecision`` with the tier, reason, matched pattern, and
    (for PROHIBITED/RESTRICTED) a deterministic refusal/confirmation answer.
    """
    q = question.strip()
    for regex, level, reason in _PATTERNS:
        if regex.search(q):
            decision = PolicyDecision(
                level=level,
                reason=reason,
                matched_pattern=regex.pattern[:60],
            )
            if level == PolicyLevel.PROHIBITED_ACTION:
                decision.deterministic_answer = (
                    f"I cannot assist with that request — it targets a safety-critical "
                    f"vehicle system and is not within my authority. ({reason}). "
                    f"Please consult a qualified vehicle technician or the vehicle manual."
                )
            elif level == PolicyLevel.RESTRICTED_ACTION:
                decision.deterministic_answer = (
                    f"I can help with that, but this action requires your explicit "
                    f"confirmation before I proceed. ({reason}). "
                    f"Please confirm on the vehicle display or repeat your request with "
                    f"'I confirm' at the start."
                )
            logger.info(
                "policy classified",
                extra={"payload": {"level": level.value, "question": q[:80], "reason": reason}},
            )
            return decision

    # Default: read-only
    return PolicyDecision(level=PolicyLevel.READ_ONLY, reason="Default classification")


def is_prohibited(question: str, intents: list[str]) -> bool:
    """Fast check — used by the safety gate before the LLM is consulted."""
    return classify(question, intents).level == PolicyLevel.PROHIBITED_ACTION
