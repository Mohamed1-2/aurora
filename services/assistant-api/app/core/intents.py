"""Intent classification for the assistant.

Intents decide which grounding silos to pull and, later, which evaluation
metrics the answer will be judged against. Kept deterministic and rule-based
so tests and offline evals are stable.
"""

from __future__ import annotations

import re

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("range", ["range", "how far", "distance", "reach", "make it", "dte", "how long will"]),
    ("driving_time", [
        "how long can i", "how long will i", "how long do i", "how long until", "how long before",
        "get there", "arrive", "arrival", "eta",
        "driving time", "keep driving", "drive for", "how many minutes",
        "minutes can i", "minutes left", "minutes do i", "time left",
        "how much driving", "how much time", "can i drive",
    ]),
    ("battery", ["battery", "soc", "state of charge", "charge level", "percent", "%"]),
    ("charging", ["charging", "plug", "fast charg", "charge rate", "dcfc", "charge port"]),
    ("tire", ["tire", "tyre", "pressure", "wheel"]),
    ("why_prediction", ["why is", "why did", "why has", "why would", "what caused", "what made", "explain the drop", "drop in soc", "why the drop", "soc drop"]),
    ("diagnostics", ["dtc", "diagnostic", "code", "check engine", "warning light", "malfunction", "p0420", "b0131", "u0100", "c0031", "p0171", "p0b36"]),
    ("thermal", ["temperature", "overheat", "coolant", "too hot", "therm"]),
    ("energy", ["consumption", "efficien", "wh/km", "wh per", "energy", "kwh", "econom"]),
    ("maintenance", ["maintenance", "servic", "replace", "interval", "rotate"]),
    ("speed", ["how fast", "speed", "velocity"]),
    ("driving_mode", ["mode", "sport", "eco"]),
    ("winter", ["winter", "cold", "ice", "snow", "freezing"]),
    ("summary", ["summary", "status", "how's my car", "how is my car", "how is the car", "car status", "health", "what do you see", "tell me about my car"]),
    ("greeting", ["hello", "hi", "hey", "good morning", "who are you"]),
]


def detect(question: str) -> list[str]:
    ql = question.lower()
    hits = []
    for name, pats in _INTENT_PATTERNS:
        if any(re.search(r"(?<!\w)" + re.escape(p) + (r"(?!\w)" if p in {"hi", "hey", "ice", "eco", "soc", "eta", "code", "mode", "sport"} else ""), ql) for p in pats):
            hits.append(name)
    return hits or ["general"]


# Signals each intent is *expected* to ground onto — used by the eval harness
# to score whether answers actually used the right live data.
EXPECTED_SIGNALS: dict[str, list[str]] = {
    "range": ["distance_to_empty_km", "battery_soc_pct", "vehicle_speed_kph"],
    "driving_time": ["distance_to_empty_km", "vehicle_speed_kph", "battery_soc_pct", "destination_remaining_km"],
    "battery": ["battery_soc_pct", "battery_temp_c", "battery_voltage_v"],
    "charging": ["charge_rate_kw", "battery_soc_pct"],
    "tire": ["tire_pressure_fl_kpa", "tire_pressure_fr_kpa", "tire_pressure_rl_kpa", "tire_pressure_rr_kpa"],
    "diagnostics": [],
    "thermal": ["coolant_temp_c", "battery_temp_c"],
    "energy": ["motor_power_kw", "regen_power_kw", "energy_session_kwh"],
    "maintenance": [],
    "speed": ["vehicle_speed_kph"],
    "driving_mode": [],
    "winter": ["outside_temp_c"],
    "summary": ["battery_soc_pct", "distance_to_empty_km", "vehicle_speed_kph"],
}

# Weight retrieval towards the most relevant docs per intent. The provenanced
# corpus (manual + supplements) is the primary authority; the committed
# knowledge-pack stays as a lowest-authority coaching fallback for guidance
# the manual never states (aero/regen tips, DTC semantics, service intervals).
INTENT_DOC_WEIGHTS: dict[str, dict[str, float]] = {
    "range": {"cupra-born-03-26-owners-manual": 1.5, "doe-cold-weather-bev": 1.5,
              "03-range-optimization.md": 3.0, "02-battery-and-charging.md": 2.0},
    "battery": {"cupra-born-03-26-owners-manual": 1.5, "cupra-born-charging-appendix": 1.5,
                "02-battery-and-charging.md": 3.0},
    "charging": {"cupra-born-charging-appendix": 2.0, "cupra-born-03-26-owners-manual": 1.5,
                 "02-battery-and-charging.md": 3.0},
    "tire": {"cupra-born-03-26-owners-manual": 2.0, "04-tires-and-wheels.md": 3.0},
    "diagnostics": {"cupra-born-03-26-owners-manual": 1.5, "05-diagnostics-and-dtc.md": 4.0},
    "thermal": {"cupra-born-03-26-owners-manual": 1.5,
                "06-emergency-and-safety.md": 3.0, "02-battery-and-charging.md": 2.0},
    "energy": {"cupra-born-03-26-owners-manual": 1.5, "doe-cold-weather-bev": 1.5,
               "03-range-optimization.md": 3.0},
    "maintenance": {"cupra-born-03-26-owners-manual": 2.0, "08-maintenance-schedule.md": 4.0},
    "winter": {"cupra-born-03-26-owners-manual": 1.5, "doe-cold-weather-bev": 2.0,
               "07-winter-and-weather.md": 4.0},
}