"""AURORA v2 — transform real telemetry into grounded instruction/context/output.

Turns normalized ``DrivingWindow`` (TUM) and ``ChargingSession`` (EPFL) records
into ``instruction / input / output`` training triples for LoRA, with full
provenance. Every output makes **only** claims the telemetry supports; derived
values (range) are flagged. Synthetic user questions and generated answers are
always labelled ``qa_type=synthetic_generated`` and never described as real
conversations.

The scheme mirrors the repo's prompt contract:
    instruction = "You are AURORA, an in-vehicle assistant grounded in the ORION-X..."
    input      = user question
    output     = grounded assistant answer
plus a ``context`` block and the provenance dict baked into each example row.

To keep the corpus diverse (no classifier template overfitting), each scenario
can draw from multiple *manually written* paraphrase templates for both the user
question and the grounded answer.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

from .base import ChargingSession, DrivingWindow

PREAMBLE = (
    "You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
    "Answer only from the provided telemetry. If data is missing, refuse honestly."
)


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _range_km(soc_pct: float) -> float:
    # Derived estimate: ~400 km nominal at 100% SOC. Explicitly derived, not measured.
    return round(400.0 * max(0.0, soc_pct) / 100.0, 1)


# ---------------------------------------------------------------------------
# TUM driving scenario generators
#
# Each generator returns a dict with: input, output, intent, safety_tier,
# anchor_signals, derived. The scenario name is supplied externally so the
# functions stay trivial/self-contained.
# ---------------------------------------------------------------------------

def _b_soc_level(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    soc = s["soc_pct"]
    speed = s.get("speed_kph", 0.0)
    if soc < 20:
        q = _pick(rng, [
            f"My battery is at {soc:.0f}%. Is that low?",
            f"I'm at {soc:.0f}% SOC. Should I worry?",
            f"Only {soc:.0f} percent left — is that getting low?",
            f"How bad is {soc:.0f}% battery right now?",
        ])
        ans = _pick(rng, [
            f"Your battery is currently at {_fmt(soc)}% SOC, which is relatively low. "
            f"At a current speed of about {_fmt(speed, 0)} km/h, I'd plan a charging stop soon.",
            f"You're down to {_fmt(soc)}% SOC — that's low. Driving at ~{_fmt(speed, 0)} km/h, "
            "I'd find a charger soon.",
        ])
    else:
        q = _pick(rng, [
            "What percentage is my battery at right now?",
            "What's my current state of charge?",
            "What percent am I at?",
            "How charged is my battery?",
            "Can you tell me my SOC?",
            "What's the battery level?",
            "How much charge do I have?",
            "What percentage is remaining?",
        ])
        ans = _pick(rng, [
            f"Your battery is at {_fmt(soc)}% SOC right now.",
            f"Your state of charge is {_fmt(soc)}%.",
            f"You're currently at {_fmt(soc)}% battery.",
        ])
    return {"input": q, "output": ans, "intent": "battery",
            "anchor_signals": ["soc_pct", "speed_kph"], "derived": {}, "safety_tier": "read_only"}


def _b_thermal_battery(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    t = s["battery_temp_c"]
    q = _pick(rng, [
        "What's my battery temperature?",
        "How hot is the battery right now?",
        "What temperature is the pack at?",
        "Is my battery running hot?",
        "What's the battery pack temp?",
        "How warm is the battery?",
        "What temperature is the HV battery?",
    ])
    if t > 42:
        ans = _pick(rng, [
            f"Your battery pack is at {_fmt(t)} °C, which is above the ~42 °C "
            "caution threshold. I'd reduce high-power driving until it cools.",
            f"The pack is reading {_fmt(t)} °C, above the safe ~42 °C mark. "
            "Ease off high power until it comes down.",
        ])
        tier = "restricted_action"
    else:
        ans = _pick(rng, [
            f"Your battery pack is currently at {_fmt(t)} °C.",
            f"The battery temperature is {_fmt(t)} °C — in the normal range.",
            f"Pack temperature is {_fmt(t)} °C right now.",
        ])
        tier = "read_only"
    return {"input": q, "output": ans, "intent": "thermal",
            "anchor_signals": ["battery_temp_c"], "derived": {}, "safety_tier": tier}


def _b_thermal_coolant(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    c = s["coolant_temp_c"]
    bat = s["battery_temp_c"]
    q = _pick(rng, [
        "How hot is the cooling system?",
        "What's the coolant temperature?",
        "Is the cooling system running hot?",
        "How warm is the coolant right now?",
        "What does the coolant temp read?",
        "How's the cooling loop temperature?",
        "What's the temperature of the cooling fluid?",
    ])
    ans = _pick(rng, [
        f"The coolant temperature at the inverter inlet is {_fmt(c)} °C "
        f"and the battery is at {_fmt(bat)} °C.",
        f"Coolant is at {_fmt(c)} °C; the pack is at {_fmt(bat)} °C.",
        f"Right now the coolant reads {_fmt(c)} °C and the battery {_fmt(bat)} °C.",
    ])
    return {"input": q, "output": ans, "intent": "thermal",
            "anchor_signals": ["coolant_temp_c", "battery_temp_c"], "derived": {}, "safety_tier": "read_only"}


def _b_speed_overview(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    speed = s["speed_kph"]
    soc = s["soc_pct"]
    q = _pick(rng, [
        "How fast am I going right now?",
        "What's my current speed?",
        "How quick am I driving at the moment?",
        "What speed am I at?",
        "How many km/h am I doing?",
        "What's the current speed of the car?",
    ])
    ans = _pick(rng, [
        f"You're travelling at about {_fmt(speed, 0)} km/h. Your battery is at {_fmt(soc)}%.",
        f"Current speed is {_fmt(speed, 0)} km/h, with {_fmt(soc)}% battery charge.",
        f"You're doing about {_fmt(speed, 0)} km/h; SOC is {_fmt(soc)}%.",
    ])
    return {"input": q, "output": ans, "intent": "speed",
            "anchor_signals": ["speed_kph", "soc_pct"], "derived": {}, "safety_tier": "read_only"}


def _b_pack_voltage(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    v = s["pack_voltage_v"]
    q = _pick(rng, [
        "What voltage is the battery pack at?",
        "What's the pack voltage?",
        "How many volts is the battery showing?",
        "What voltage does the high-voltage pack read?",
        "What's the HV bus voltage?",
        "What voltage is the battery at?",
    ])
    ans = _pick(rng, [
        f"Your high-voltage pack is sitting at about {_fmt(v, 0)} V.",
        f"The pack voltage is about {_fmt(v, 0)} V.",
        f"You're at about {_fmt(v, 0)} V on the high-voltage bus.",
    ])
    return {"input": q, "output": ans, "intent": "battery",
            "anchor_signals": ["pack_voltage_v"], "derived": {}, "safety_tier": "read_only"}


def _b_cold_range(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    soc = s["soc_pct"]
    outside = s["outside_temp_c"]
    rng_km = _range_km(soc)
    q = _pick(rng, [
        "How far can I drive with the current charge in this cold weather?",
        "What's my estimated range right now?",
        "How many kilometres do I have left?",
        "Can you estimate how far I can go?",
        "What's my remaining driving range?",
        "How far will I get on this charge?",
        "What range do I have left?",
    ])
    ans = _pick(rng, [
        f"At {_fmt(soc)}% SOC, a rough range estimate is about {_fmt(rng_km, 0)} km. "
        "This is an estimate; cold temperatures reduce real-world range.",
        f"Roughly {_fmt(rng_km, 0)} km considering your {_fmt(soc)}% charge — "
        "cold weather cuts that in practice.",
    ])
    return {"input": q, "output": ans, "intent": "range",
            "anchor_signals": ["soc_pct", "outside_temp_c"],
            "derived": {"range_km": {"value": rng_km, "note": "derived from SOC × nominal 400km/100%, not measured"}},
            "safety_tier": "read_only"}


def _b_vehicle_status_summary(w: DrivingWindow, rng: random.Random) -> dict:
    s = w.signals
    soc = s["soc_pct"]
    speed = s["speed_kph"]
    bat = s["battery_temp_c"]
    outside = s["outside_temp_c"]
    q = _pick(rng, [
        "Give me a quick status of the vehicle.",
        "What's the vehicle status right now?",
        "How is my car doing?",
        "Can you give me a summary of the car state?",
        "What's happening with the car right now?",
        "Give me the current vehicle summary.",
        "What's the state of the vehicle?",
        "How's everything looking with the car?",
    ])
    ans = _pick(rng, [
        f"Here's a snapshot: battery {_fmt(soc)}% SOC, speed {_fmt(speed, 0)} km/h, "
        f"battery temp {_fmt(bat)} °C, outside {_fmt(outside)} °C.",
        f"Current state — SOC {_fmt(soc)}%, {_fmt(speed, 0)} km/h, "
        f"pack {_fmt(bat)} °C, ambient {_fmt(outside)} °C.",
    ])
    return {"input": q, "output": ans, "intent": "summary",
            "anchor_signals": ["soc_pct", "speed_kph", "battery_temp_c", "outside_temp_c"],
            "derived": {}, "safety_tier": "read_only"}


_DRIVING_SCENARIOS: dict[str, tuple[set[str], object]] = {
    "soc_level": ({"soc_pct"}, _b_soc_level),
    "thermal_battery": ({"battery_temp_c"}, _b_thermal_battery),
    "thermal_coolant": ({"coolant_temp_c", "battery_temp_c"}, _b_thermal_coolant),
    "speed_overview": ({"speed_kph", "soc_pct"}, _b_speed_overview),
    "pack_voltage": ({"pack_voltage_v"}, _b_pack_voltage),
    "cold_range": ({"soc_pct", "outside_temp_c"}, _b_cold_range),
    "vehicle_status_summary": ({"soc_pct", "speed_kph", "battery_temp_c", "outside_temp_c"}, _b_vehicle_status_summary),
}


def build_driving_examples(
    windows: Iterable[DrivingWindow],
    rng: random.Random,
    cap_per_window: int = 1,
) -> list[dict]:
    out: list[dict] = []
    for w in windows:
        eligible = [name for name, (need, _fn) in _DRIVING_SCENARIOS.items()
                    if need.issubset(w.signals)]
        if not eligible:
            continue
        for _ in range(cap_per_window):
            name = rng.choice(eligible)
            fn = _DRIVING_SCENARIOS[name][1]
            ex = fn(w, rng)
            prov = w.provenance.to_dict()
            prov["qa_type"] = "synthetic_generated"
            out.append(example_row(
                instruction=PREAMBLE,
                input=ex["input"],
                output=ex["output"],
                context={k: w.signals[k] for k in ex["anchor_signals"]},
                intent=ex["intent"],
                safety_tier=ex.get("safety_tier", "read_only"),
                provenance=prov,
                derived=ex.get("derived", {}),
                scenario=name,
            ))
    return out


# ---------------------------------------------------------------------------
# EPFL charging scenario generators
# ---------------------------------------------------------------------------

def _c_energy_received(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    energy_kwh = round(s["energy_charged_kwh"], 2)
    q = _pick(rng, [
        "How much energy have I received during this charging session?",
        "How many kWh did I get from this charge?",
        "What was the total energy delivered?",
        "How much energy did this session add?",
    ])
    ans = _pick(rng, [
        f"In this charging session you received about {energy_kwh} kWh.",
        f"This session delivered about {energy_kwh} kWh of energy.",
        f"You gained roughly {energy_kwh} kWh from this charge.",
    ])
    return {"input": q, "output": ans, "intent": "energy",
            "anchor_signals": ["energy_charged_kwh"], "derived": {}, "safety_tier": "read_only"}


def _c_session_duration(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    stay = s["stay_min"]
    q = _pick(rng, [
        "How long has this charging session lasted?",
        "How many minutes was I plugged in?",
        "What was the duration of my charging session?",
        "How long was this charge?",
    ])
    ans = _pick(rng, [
        f"This charging session lasted about {_fmt(stay, 0)} minutes.",
        f"You were charging for roughly {_fmt(stay, 0)} minutes.",
        f"The session ran about {_fmt(stay, 0)} minutes.",
    ])
    return {"input": q, "output": ans, "intent": "charging",
            "anchor_signals": ["stay_min"], "derived": {}, "safety_tier": "read_only"}


def _c_power_lower_than_expected(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    power = s["max_power_kw"]
    preq = s["max_requested_power_kw"]
    controlled = s["controlled_session"]
    q = _pick(rng, [
        "Why is my charging power lower than expected?",
        "Why am I not getting full charging power?",
        "Why is the charging rate lower than it should be?",
        "Why is my charge slower than expected?",
    ])
    if controlled:
        ans = _pick(rng, [
            f"Your session was power-limited by the charging station: max power here "
            f"was about {_fmt(power)} kW while your vehicle requested up to about "
            f"{_fmt(preq)} kW. Shared station capacity or station limits reduced it.",
            f"The station capped this session at about {_fmt(power)} kW even though the "
            f"car asked for up to {_fmt(preq)} kW — station or shared-capacity limits.",
        ])
    else:
        ans = _pick(rng, [
            f"Your session reached about {_fmt(power)} kW, matching what the vehicle requested.",
            f"You hit about {_fmt(power)} kW, which is what the car requested — no station cap.",
        ])
    return {"input": q, "output": ans, "intent": "charging",
            "anchor_signals": ["max_power_kw", "max_requested_power_kw", "controlled_session"],
            "derived": {}, "safety_tier": "read_only"}


def _c_soc_arrival(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    soc = s["soc_arrival_pct"]
    q = _pick(rng, [
        "What was my state of charge when I started charging?",
        "What percent was my battery at the start of charging?",
        "What was my SOC when I plugged in?",
        "How charged was I when the session began?",
    ])
    ans = _pick(rng, [
        f"When you started charging, your battery was at about {_fmt(soc, 0)}% SOC.",
        f"You arrived at roughly {_fmt(soc, 0)}% SOC.",
        f"At session start your SOC was about {_fmt(soc, 0)}%.",
    ])
    return {"input": q, "output": ans, "intent": "charging",
            "anchor_signals": ["soc_arrival_pct"], "derived": {}, "safety_tier": "read_only"}


def _c_soc_departure(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    soc = s["soc_departure_pct"]
    q = _pick(rng, [
        "What will my battery level be when I leave, given this session?",
        "What SOC will I have when I unplug?",
        "How charged will the battery be at the end of charging?",
        "What percent should I have when this session finishes?",
    ])
    ans = _pick(rng, [
        f"This session would bring you to about {_fmt(soc, 0)}% SOC at departure.",
        f"You'd leave at roughly {_fmt(soc, 0)}% SOC.",
        f"At the end of this session expect about {_fmt(soc, 0)}%.",
    ])
    return {"input": q, "output": ans, "intent": "charging",
            "anchor_signals": ["soc_departure_pct"], "derived": {}, "safety_tier": "read_only"}


def _c_session_summary(cs: ChargingSession, rng: random.Random) -> dict:
    s = cs.summary
    q = _pick(rng, [
        "Summarize my charging session.",
        "Give me a summary of this charge.",
        "What happened during my charging session?",
        "Can you summarize this charging session for me?",
    ])
    ans = _pick(rng, [
        f"You arrived at about {_fmt(s['soc_arrival_pct'], 0)}% SOC and departed at "
        f"{_fmt(s['soc_departure_pct'], 0)}%. The session lasted about "
        f"{_fmt(s['stay_min'], 0)} minutes and delivered about "
        f"{round(s['energy_charged_kwh'], 2)} kWh.",
        f"Session: {_fmt(s['stay_min'], 0)} min, {round(s['energy_charged_kwh'], 2)} kWh, "
        f"SOC went from {_fmt(s['soc_arrival_pct'], 0)}% to {_fmt(s['soc_departure_pct'], 0)}%.",
    ])
    return {"input": q, "output": ans, "intent": "summary",
            "anchor_signals": ["soc_arrival_pct", "soc_departure_pct", "stay_min", "energy_charged_kwh"],
            "derived": {}, "safety_tier": "read_only"}


_CHARGING_SCENARIOS: dict[str, tuple[set[str], object]] = {
    "energy_received": ({"energy_charged_kwh"}, _c_energy_received),
    "session_duration": ({"stay_min"}, _c_session_duration),
    "power_lower_than_expected": ({"max_power_kw", "max_requested_power_kw", "controlled_session"}, _c_power_lower_than_expected),
    "soc_arrival": ({"soc_arrival_pct"}, _c_soc_arrival),
    "soc_departure": ({"soc_departure_pct"}, _c_soc_departure),
    "charging_session_summary": ({"soc_arrival_pct", "soc_departure_pct", "stay_min", "energy_charged_kwh"}, _c_session_summary),
}


def build_charging_examples(
    sessions: Iterable[ChargingSession],
    rng: random.Random,
    cap_per_session: int = 2,
) -> list[dict]:
    out: list[dict] = []
    for cs in sessions:
        if not cs.measurements:
            continue
        eligible = [name for name, (need, _fn) in _CHARGING_SCENARIOS.items()
                    if need.issubset(cs.summary)]
        if not eligible:
            continue
        for _ in range(cap_per_session):
            name = rng.choice(eligible)
            fn = _CHARGING_SCENARIOS[name][1]
            ex = fn(cs, rng)
            prov = cs.provenance.to_dict()
            prov["qa_type"] = "synthetic_generated"
            out.append(example_row(
                instruction=PREAMBLE,
                input=ex["input"],
                output=ex["output"],
                context={k: cs.summary[k] for k in ex["anchor_signals"]},
                intent=ex["intent"],
                safety_tier=ex.get("safety_tier", "read_only"),
                provenance=prov,
                derived=ex.get("derived", {}),
                scenario=name,
            ))
    return out


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def example_row(
    *,
    instruction: str,
    input: str,
    output: str,
    context: dict,
    intent: str,
    safety_tier: str,
    provenance: dict,
    derived: dict,
    scenario: str,
) -> dict:
    return {
        "id": _stable_id(provenance, input),
        "instruction": instruction,
        "input": input,
        "output": output,
        "context": context,
        "intent": intent,
        "safety_tier": safety_tier,
        "derived": derived,
        "scenario": scenario,
        "provenance": provenance,
    }


def _stable_id(prov: dict, question: str) -> str:
    raw = f"{prov.get('source_dataset')}::{prov.get('source_record')}::{prov.get('session_id') or prov.get('episode_id')}::{question}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def to_jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, default=str) for r in rows) + "\n"
