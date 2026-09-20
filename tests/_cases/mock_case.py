"""Behavioural checks for the grounded mock provider, run from the gateway process."""

import sys

sys.path.insert(0, "services/llm-gateway")

from app.providers.mock import MockProvider, answer, _HANDLERS, _intents

_docs = [
    {"doc_id": "03-range-optimization.md", "section": "Efficient Highway Driving", "title": "Range Optimization", "content": "Aero drag dominates above 90 km/h. 120 vs 100 km/h costs ~30% more per km. Precondition while plugged in.", "score": 6.4, "source": "03-range-optimization.md#Efficient Highway Driving"},
    {"doc_id": "05-diagnostics-and-dtc.md", "section": "Reading the codes", "title": "Diagnostics and Trouble Codes", "content": "Active DTCs are stored with severity and freeze-frame data. Warnings should be serviced promptly.", "score": 5.1, "source": "05-diagnostics-and-dtc.md#Reading the codes"},
    {"doc_id": "08-maintenance-schedule.md", "section": "Service intervals", "title": "Maintenance Schedule", "content": "Service every 24 months or 30,000 km, whichever comes first. Brake fluid every 2 years.", "score": 4.4, "source": "08-maintenance-schedule.md#Service intervals"},
    {"doc_id": "02-battery-and-charging.md", "section": "Charging overview", "title": "Battery and Charging", "content": "AC charging to 22 kW, DC fast charging to 150 kW. Stop at 80% by default.", "score": 3.3, "source": "02-battery-and-charging.md#Charging overview"},
]

_telemetry = {
    "signals": {
        "battery_soc_pct": 68.0,
        "distance_to_empty_km": 250.9,
        "vehicle_speed_kph": 90.0,
        "motor_power_kw": 18.5,
        "regen_power_kw": -9.2,
        "charge_rate_kw": 0.0,
        "coolant_temp_c": 91.6,
        "battery_temp_c": 31.2,
        "outside_temp_c": 16.0,
        "tire_pressure_fl_kpa": 242.0,
    },
    "mean": {"battery_soc_pct": 68.0, "distance_to_empty_km": 250.9},
    "trending": {"distance_to_empty_km": -0.4},
    "mode": "driving",
    "charging": False,
    "warnings": [],
    "dtcs": [{"code": "P0B36", "severity": "minor"}],
}

_INTENTS = [
    ("How far can I go?", "range"),
    ("What is my state of charge?", "battery"),
    ("Am I plugged in and charging?", "charging"),
    ("Check my tyre pressures", "tire"),
    ("Any diagnostic trouble codes?", "dtc"),
    ("Is the battery too hot?", "temperature"),
    ("How efficiently am I driving?", "consumption"),
    ("When is my next service?", "maintenance"),
    ("How fast am I going?", "speed"),
    ("It is cold and icy, any tips?", "winter"),
    ("Give me a status summary", "status"),
    ("Hello there", "hello"),
    ("Tell me a joke", "general"),
]

_ANSWERS = [
    ("What is my remaining range?", "251 km"),
    ("Am I plugged in and charging?", "0 kW"),
    ("How efficiently am I driving?", "18 kW"),
    ("Is the coolant ok?", "92 C"),
    ("What is the battery temperature?", "31 C"),
]


def main():
    fails = []

    for q, expected in _INTENTS:
        if expected not in _intents(q):
            fails.append(f"intent {q!r} -> {_intents(q)} (wanted {expected})")

    cold = _intents("It's cold and icy today -- any range tips for me?")
    if "winter" not in cold:
        fails.append("cold range-tips should detect winter intent")

    for q, needle in _ANSWERS:
        out = answer(q, _telemetry, _docs)
        if needle not in out:
            fails.append(f"{q!r} answered missing {needle!r}: {out[:80]}")
        if "【" not in out:
            fails.append(f"{q!r} missing citation markers")

    if "P0B36" not in answer("Any diagnostic trouble codes?", _telemetry, _docs):
        fails.append("P0B36 not surfaced")

    refused = answer("Who wrote War and Peace?", _telemetry, _docs).lower()
    if not any(k in refused for k in ("don't have", "can't answer", "no reliable data")):
        fails.append("unanswerable question not refused")

    _why_block = (
        "Why did my SOC drop so fast?\n\n"
        "## PREDICTION EXPLANATION (deterministic — quote only, never invent numbers)\n"
        "- Predicted SOC drop 1.64 SOC pts (physics 1.45 SOC pts + ML adjustment 0.19 SOC pts; model v2-ws1-r1; capacity 58.0 kWh).\n"
        "- Status: active. Fallback: none.\n"
        "- Change vs earlier window: 1.39 SOC pts -> 1.64 SOC pts (delta 0.25 SOC pts).\n"
        "- Top causes (ranked): rolling +0.77; aero +0.65; ML regen_decel_energy_kwh +0.35; "
        "ML speed_kph_std +0.12; physics_soc_drop_pct +0.09.\n"
        "- ML model bias 1.05 SOC pts is a model attribute, not a physical cause."
    )
    why = answer("Why did my SOC drop so fast?", _telemetry, _docs, full_user=_why_block)
    for needle in ("1.64 SOC pts", "physics 1.45", "rolling +0.77", "aero +0.65", "v2-ws1-r1", "by 0.25 SOC pts"):
        if needle not in why:
            fails.append(f"why answer missing {needle!r}: {why[:120]}")
    if "compared with the earlier window" not in why.lower():
        fails.append("why answer should cite the delta vs the earlier window")

    no_block = answer("Why did my SOC drop so fast?", _telemetry, _docs)
    if "1.64" in no_block:
        fails.append("why without a block must not fabricate prediction numbers")

    malformed = answer(
        "Why did my SOC drop so fast?", _telemetry, _docs,
        full_user="Why did my SOC drop so fast?\n\n## PREDICTION EXPLANATION\n- Status: active, but no prediction line.\n",
    )
    if "unverified" not in malformed.lower():
        fails.append("malformed block should refuse to quote unverified numbers")

    prov = MockProvider()
    from vehicle_ai.schemas import GatewayRequest, Message

    async def _complete():
        return await prov.complete(
            GatewayRequest(
                messages=[Message(role="user", content="What is my remaining range?")],
                grounding={"telemetry": {"signals": {"distance_to_empty_km": 250.9, "battery_soc_pct": 68.0}, "mean": {}, "mode": "driving", "charging": False, "warnings": [], "dtcs": []}, "documents": _docs, "events": []},
            )
        )

    import asyncio

    res = asyncio.run(_complete())
    if "251 km" not in res.content:
        fails.append("MockProvider.complete pipeline broken")

    print(f"MOCK_CHECKS passed={len(_INTENTS) + len(_ANSWERS) + 6 - len(fails)} failed={len(fails)}")
    if fails:
        print("FAILED:", *fails, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()