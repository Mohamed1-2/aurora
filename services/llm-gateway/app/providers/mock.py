"""Grounded local provider — AURORA's offline / no-cloud path.

This provider answers from the *actual* grounding it is given (live telemetry
windows, retrieved KB documents, active DTCs). It is deterministic, auditable
and fast, which makes it the workhorse of CI, offline evaluation, and the
safe default when cloud providers are unreachable. It deliberately refuses
questions it cannot ground — the same honesty contract we train LLMs toward.
"""

from __future__ import annotations

import re
import time
from typing import Any

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.schemas import GatewayRequest

from .base import BaseProvider, ProviderResult, extract_grounding

logger = get_logger("aurora.gateway.mock")

_CITE = "【{}：{}】"


def _sig(telemetry: dict, name: str, default: float | None = None) -> float | None:
    v = telemetry.get("signals", {}).get(name)
    if v is None and "mean" in telemetry:
        v = telemetry["mean"].get(name)
    if v is None:
        return default
    try:
        f = float(v)
        return f if f == f else default  # NaN guard
    except (TypeError, ValueError):
        return default


def _fmt(telemetry: dict, name: str, unit: str = "", default: str = "--") -> str:
    v = _sig(telemetry, name)
    if v is None:
        return default
    return f"{v:.0f}{unit}"


def _trend(telemetry: dict, name: str) -> float | None:
    v = telemetry.get("trending", {}).get(name)
    return float(v) if isinstance(v, (int, float)) else None


class MockProvider(BaseProvider):
    name = "mock"
    models = ["aurora-titan-e-local"]

    def is_configured(self) -> bool:
        return True

    async def complete(self, req: GatewayRequest) -> ProviderResult:
        start = time.perf_counter()
        g = extract_grounding(req)
        raw_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        # The assistant appends grounding context after a "\n\n## " marker —
        # intent detection must run on the driver's question only, never the
        # context (which can accidentally match keywords, e.g. "hi" inside JSON).
        question = raw_user.split("\n\n## ")[0] if "\n\n## " in raw_user else raw_user
        content = answer(question, g["telemetry"], g["documents"], full_user=raw_user)
        latency = (time.perf_counter() - start) * 1000
        return ProviderResult(
            content=content,
            reasoning="grounded-offline-provider",
            prompt_tokens=len(raw_user) // 4,
            completion_tokens=len(content) // 4,
            latency_ms=latency,
            model=self.models[0],
            region="local",
            cost_per_1k_in=0.0,
            cost_per_1k_out=0.0,
        )


# ---------------------------------------------------------------------------
# Intent detection + template generation
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("range", ["range", "how far", "how long will", "distance i can", "reach", "make it to", "dte"]),
    ("battery", ["battery", "soc", "state of charge", "charge level", "% left", "percent left"]),
    ("charging", ["charging", "vplugin", "plug in", "charge port", "fast charg", "dcfc"]),
    ("tire", ["tire", "tyre", "pressure of", "wheel"]),
    ("dtc", ["dtc", "diagnostic", "code", "check engine", "warning light", "malfunction", "p0420", "u0100", "c0031", "p0171", "b0131", "p0b36"]),
    ("temperature", ["temperature", "overheat", "overheat", "coolant", "too hot", "battery temp", "therm"]),
    ("consumption", ["consumption", "efficien", "wh/km", "wh per", "energy use", "econ", "kwh", "energy used"]),
    ("maintenance", ["maintenance", "servic", "replace", "interval", "rotate"]),
    ("speed", ["how fast", "speed", "going", "current velocity"]),
    ("driving_mode", ["mode", "sport", "eco mode", "driving mode"]),
    ("winter", ["winter", "cold", "ice", "snow", "freezing"]),
    ("status", ["status", "summary", "how's my car", "how is my car", "health", "what do you see", "can you see", "what's happening"]),
    ("hello", ["hello", "hi", "hey", "good morning", "good evening", "who are you", "what can you do"]),
]


def _intents(q: str) -> list[str]:
    ql = q.lower()
    hits = []
    for name, pats in _INTENT_PATTERNS:
        if any(re.search(r"(?<!\w)" + re.escape(p) + (r"(?!\w)" if p in {"hi", "hey", "ice", "eco", "soc", "eta", "code", "mode", "sport"} else ""), ql) for p in pats):
            hits.append(name)
    return hits or ["general"]


def answer(question: str, telemetry: dict, documents: list[dict], full_user: str = "") -> str:
    if not telemetry.get("signals") and not telemetry.get("mean"):
        return (
            "I don't have live telemetry to ground that on right now — the vehicle stream is disconnected. "
            "I can still answer manual-style questions (maintenance, tyres, battery care, emergency checks) "
            "from the owners manual." + _cite(documents, "owners") + _cite(documents, "emergency")
        )
    # WS3: when the caller supplied a deterministic prediction explanation block,
    # quote it verbatim — never let an offline template invent causes or numbers.
    why = _h_why(full_user)
    if why:
        return why
    intents = _intents(question)
    for intent in _DISPATCH_ORDER:
        if _HANDLERS.get(intent) and intent in intents:
            try:
                out = _HANDLERS[intent](telemetry, documents, question)
            except Exception as exc:  # noqa: BLE001 — never let a template crash the gateway
                logger.warning("mock handler failed", extra={"payload": {"intent": intent, "error": str(exc)}})
                out = "" if intent == "general" else _refuse(question, telemetry)
            if out:
                return out
    return _refuse(question, telemetry)


def _cite(docs: list[dict], needle: str) -> str:
    for d in docs:
        if needle in d.get("source", "") or needle.lower() in d.get("title", "").lower():
            return _CITE.format(d["title"], d["section"])
    return ""


def _refuse(question: str, telemetry: dict) -> str:
    return (
        "I don't have reliable data to answer that from the vehicle stream or the knowledge base. "
        "I can help with range, battery, tyres, charging, diagnostics, temperatures, maintenance and "
        f"driving tips for this ORION-X. Right now, for example, battery is at {_fmt(telemetry, 'battery_soc_pct', '%')} "
        f"and estimated range is {_fmt(telemetry, 'distance_to_empty_km', ' km')}. Ask me one of those."
    )


def _h_why(full_user: str) -> str:
    """WS3 offline path: quote the deterministic ``## PREDICTION EXPLANATION`` block.

    Returns an empty string when no block is present (normal dispatch resumes);
    when present, every figure is copied verbatim from the block — the provider
    must never invent causes or numbers, mirroring the LLM's quote-only rule.
    """
    marker = "## PREDICTION EXPLANATION"
    idx = full_user.find(marker)
    if idx < 0:
        return ""
    block = full_user[idx:]
    predicted = re.search(
        r"Predicted SOC drop ([\d.\-]+) SOC pts \(physics ([\d.\-]+)(?: SOC pts)? \+ ML adjustment "
        r"([\d.\-]+)(?: SOC pts)?; model ([^;]+); capacity ([\d.\-]+) kWh\)",
        block,
    )
    if not predicted:
        return (
            "I have a deterministic prediction explanation for this, but its figures look malformed — "
            "I won't quote unverified numbers."
        )
    drop, physics, ml, model, cap = (m.strip() for m in predicted.groups())
    parts = [
        f"From the deterministic SOC-drop model ({model}; {cap} kWh usable pack), the prediction for "
        f"the current driving window is a drop of about {drop} SOC pts — physics {physics} + ML "
        f"adjustment {ml}."
    ]
    d = re.search(r"Change vs earlier window: [\d.\-]+(?: SOC pts)? -> [\d.\-]+(?: SOC pts)? \(delta ([\d.\-]+) SOC pts\)", block)
    if d:
        parts.append(f"Compared with the earlier window, the predicted drop changed by {d.group(1).strip()} SOC pts.")
    tc = re.search(r"^-\s*Top causes \(ranked\): (.+)$", block, re.M)
    if tc:
        causes = [c.strip().rstrip(".") for c in tc.group(1).split(";") if c.strip()]
        parts.append("Ranked causes (computed in code, quoted exactly): " + "; ".join(causes) + ".")
    else:
        parts.append("No ranked causes are available for this explanation.")
    st = re.search(r"^-\s*Status: ([^\n]+)", block, re.M)
    if st:
        _status = st.group(1).strip().rstrip(".")
        if "physics_only_fallback" in block:
            _status += " — physics-only (ML model unavailable; nothing invented)"
        parts.append(f"Status: {_status}.")
    return " ".join(parts)


def _common_prefix(telemetry: dict) -> str:
    return (
        f"Vehicle state: SOC {_fmt(telemetry, 'battery_soc_pct', '%')}, "
        f"range {_fmt(telemetry, 'distance_to_empty_km', ' km')}, "
        f"speed {_fmt(telemetry, 'vehicle_speed_kph', ' km/h')}, "
        f"bat temp {_fmt(telemetry, 'battery_temp_c', ' C')}."
    )


def _h_status(telemetry: dict, docs: list[dict], q: str) -> str:
    soc = _sig(telemetry, "battery_soc_pct")
    dte = _sig(telemetry, "distance_to_empty_km")
    speed = _sig(telemetry, "vehicle_speed_kph")
    bat = _sig(telemetry, "battery_temp_c")
    coolant = _sig(telemetry, "coolant_temp_c")
    tires = [
        ("FL", _sig(telemetry, "tire_pressure_fl_kpa")),
        ("FR", _sig(telemetry, "tire_pressure_fr_kpa")),
        ("RL", _sig(telemetry, "tire_pressure_rl_kpa")),
        ("RR", _sig(telemetry, "tire_pressure_rr_kpa")),
    ]
    tires_txt = ", ".join(f"{k} {v:.0f}" for k, v in tires if v) + " kPa"
    dtcs = telemetry.get("dtcs", [])
    warnings = telemetry.get("warnings", [])
    lines = [
        "Here's a live snapshot of your ORION-X:",
        f"- Battery: **{soc:.0f}%** ({'charging' if telemetry.get('charging') else 'not charging'}), estimated range **{dte:.0f} km**",
        f"- Driving: {speed:.0f} km/h, battery {bat:.0f} C, coolant {coolant:.0f} C",
        f"- Tyre pressures: {tires_txt}",
    ]
    if dtcs:
        codes = ", ".join(d["code"] for d in dtcs)
        lines.append(f"- Active trouble codes: **{codes}** — see the diagnostics section.")
        lines.append(_cite(docs, "diagnostics"))
    if warnings:
        lines.append(f"- Driver warnings: {'; '.join(w.get('message', w.get('kind', '')) for w in warnings)}")
    lines.append("\nWhat would you like to dig into — range, charging, tyres, or diagnostics?")
    return "\n".join(lines)


def _h_hello(telemetry: dict, docs: list[dict], q: str) -> str:
    return "Hi — I'm AURORA, your in-vehicle assistant, grounded on this vehicle's live telemetry and the ORION-X knowledge base.\n\n" + _h_status(telemetry, docs, q)


def _h_range(telemetry: dict, docs: list[dict], q: str) -> str:
    dte = _sig(telemetry, "distance_to_empty_km")
    soc = _sig(telemetry, "battery_soc_pct")
    speed = _sig(telemetry, "vehicle_speed_kph")
    trend = _trend(telemetry, "distance_to_empty_km")
    txt = [f"Estimated remaining range is **{dte:.0f} km** at {soc:.0f}% charge."]
    if speed and speed > 115:
        txt.append(f"You're at {speed:.0f} km/h right now — above ~115 km/h aero drag climbs sharply, so easing to 100-110 km/h would stretch range.")
    elif speed and speed > 0:
        txt.append(f"At the current steady pace near {speed:.0f} km/h you're in a range-efficient regime.")
    if trend is not None and trend < 0:
        txt.append(f"Range is trending down about {abs(trend):.1f} km/min — typical for current driving.")
    tx = _trend(telemetry, "distance_to_empty_km")
    if soc is not None and soc < 15:
        txt.append(f"⚠ **Critical charge ({soc:.0f}%).** Plan a charging stop: a 150 kW DC unit adds roughly 10-80% in ~33 minutes.")
    txt.append(
        "To maximise range on a long highway drive: hold a steady 100-110 km/h (aero drag dominates above 90 km/h and "
        "120 vs 100 km/h costs ~30% more per km), use Eco mode, let regen do the braking, and pre-condition the cabin while plugged in."
        + _cite(docs, "range") + _cite(docs, "battery")
    )
    return " ".join(txt)


def _h_battery(telemetry: dict, docs: list[dict], q: str) -> str:
    soc = _sig(telemetry, "battery_soc_pct")
    bat = _sig(telemetry, "battery_temp_c")
    volt = _sig(telemetry, "battery_voltage_v")
    charging = telemetry.get("charging")
    txt = [
        f"Battery is at **{soc:.0f}%** ({volt:.0f} V pack voltage), {'plugged in — charging' if charging else 'not charging'}, pack temperature **{bat:.0f} C**."
    ]
    if bat and bat > 42:
        txt.append("⚠ **Pack is above 42 C** — fast charging is not recommended until it cools; reduce load and let thermal management work." + _cite(docs, "battery"))
    elif bat and bat > 35:
        txt.append("Pack is warm but within the safe 20-35 C operating band.")
    if soc and soc < 18:
        txt.append("Below 18% the range-extender engine engages to top up the pack during drives." + _cite(docs, "overview"))
    txt.append("Tip: keep the pack in its optimal thermal band and avoid storing the vehicle above 60% SOC for long periods to preserve long-term capacity." + _cite(docs, "battery"))
    return " ".join(txt)


def _h_charging(telemetry: dict, docs: list[dict], q: str) -> str:
    rate = _sig(telemetry, "charge_rate_kw")
    soc = _sig(telemetry, "battery_soc_pct")
    if telemetry.get("charging"):
        return f"You're currently charging at **{rate:.0f} kW**, at **{soc:.0f}%**. The vehicle stops at your configured target (default 80%). AC home charging refills overnight; a 150 kW DC unit takes ~33 min for 10-80%." + _cite(docs, "battery")
    soc_now = _sig(telemetry, "battery_soc_pct")
    return (
        f"You're not plugged in — SOC is **{soc_now:.0f}%** and charge rate is **{rate:.0f} kW**, so no charging is happening right now. "
        "The vehicle supports AC charging up to 22 kW and DC fast charging to 150 kW. "
        "For public DC units, plug in above 10% SOC — the pack throttles charge below that to protect cells." + _cite(docs, "battery")
    )


def _h_tire(telemetry: dict, docs: list[dict], q: str) -> str:
    tire_map = [
        ("front-left", "tire_pressure_fl_kpa"),
        ("front-right", "tire_pressure_fr_kpa"),
        ("rear-left", "tire_pressure_rl_kpa"),
        ("rear-right", "tire_pressure_rr_kpa"),
    ]
    val = [(label, _sig(telemetry, k)) for label, k in tire_map]
    txt = ["Tyre pressures right now:"]
    low = []
    for label, v in val:
        if v is None:
            continue
        flag = ""
        if v < 200:
            flag = " — ⚠ **too low, unsafe for continued highway driving**"
            low.append(label)
        elif v < 210:
            flag = " — ⚠ low, inflate soon"
            low.append(label)
        txt.append(f"- {label}: **{v:.0f} kPa**{flag}")
    txt.append("Nominal is 240 kPa; cold mornings can dip ~10-14 kPa per 10 C drop, which is normal." + _cite(docs, "tyres"))
    if low:
        txt.append(f"Recommendation: verify all four corners and inspect for leaks/punctures before further driving; keep a portable inflator handy." + _cite(docs, "emergency"))
    return "\n".join(txt)


def _h_dtc(telemetry: dict, docs: list[dict], q: str) -> str:
    dtcs = telemetry.get("dtcs", [])
    warnings = telemetry.get("warnings", [])
    if not dtcs:
        # Question may reference a specific code hypothetically.
        if re.search(r"p0b36", q.lower()):
            return "P0B36 is a high-voltage issue: hybrid battery pack voltage inconsistent. Charging should stop and a certified technician should inspect the vehicle — this is safety-critical." + _cite(docs, "diagnostics")
        if re.search(r"c0031", q.lower()):
            return "C0031 is a braking-system fault: left front wheel speed sensor circuit failure, which degrades ABS and stability control. Drive cautiously, expect longer stopping distances on low-grip surfaces, and service promptly." + _cite(docs, "diagnostics")
        if re.search(r"b0131", q.lower()):
            return "B0131 is a restraint fault: the driver frontal airbag circuit is short to ground. The airbag may not deploy — treat this as safety-critical and do not drive until it's resolved." + _cite(docs, "diagnostics")
        if re.search(r"p0420", q.lower()):
            return "P0420 means catalyst efficiency below threshold — usually ageing or misfueling. Driving is generally fine but plan a service check within ~500 km; emissions and economy suffer." + _cite(docs, "diagnostics")
        if re.search(r"u0100", q.lower()):
            return "U0100 is a CAN-bus communication drop (lost comm with the ECM/PCM). It can cause instrument warnings and degraded control — service promptly and limit driving." + _cite(docs, "diagnostics")
        if re.search(r"p0171", q.lower()):
            return "P0171 means the mixture is running lean, usually a vacuum leak or sensors. Driving is possible; service is recommended." + _cite(docs, "diagnostics")
        return "No active diagnostic trouble codes in the telemetry stream right now. All monitored systems are reporting in range."
    lines = [f"There are **{len(dtcs)} active trouble code(s)** on this vehicle:"]
    for d in dtcs:
        sev = "⚠ safety-critical" if d.get("severity") == "critical" else "advisory"
        lines.append(f"- **{d['code']}** — {d.get('description', '')} ({sev}).")
    codes = {d["code"] for d in dtcs}
    if {"B0131", "C0031", "P0B36"} & codes:
        lines.append("Some of these are safety-critical (restraints, braking or HV). Please schedule immediate service." )
    else:
        lines.append("These are advisory — continued driving is generally safe, but schedule a service check." )
    lines.append(_cite(docs, "diagnostics"))
    if warnings:
        lines.append("Also active right now: " + "; ".join(w.get("message", "") for w in warnings))
    return " ".join(lines)


def _h_temperature(telemetry: dict, docs: list[dict], q: str) -> str:
    coolant = _sig(telemetry, "coolant_temp_c")
    bat = _sig(telemetry, "battery_temp_c")
    txt = [f"Coolant is **{coolant:.0f} C** and the high-voltage battery is **{bat:.0f} C**."]
    if coolant and coolant > 110:
        txt.append("⚠ **Coolant over 110 C** — pull over as soon as safe, switch off and wait for it to fall. Do not open the reservoir while hot." + _cite(docs, "emergency"))
    elif coolant and coolant > 103:
        txt.append("Coolant is above 103 C — reduce load now: Eco mode, ease speed, open windows to cut HVAC draw." + _cite(docs, "emergency"))
    elif coolant and coolant < 60:
        txt.append("Coolant is low — normal right after a cold start; it will settle near 88 C.")
    if bat and bat > 42:
        txt.append("⚠ Battery at high temperature: avoid DC fast charging until the pack cools." + _cite(docs, "battery"))
    elif bat:
        txt.append("Battery is comfortably inside the optimal 20-35 C band." + _cite(docs, "battery"))
    return " ".join(txt)


def _h_consumption(telemetry: dict, docs: list[dict], q: str) -> str:
    motor = _sig(telemetry, "motor_power_kw")
    regen = _sig(telemetry, "regen_power_kw")
    energy = _sig(telemetry, "energy_session_kwh")
    trip = _sig(telemetry, "trip_distance_km")
    eco = _sig(telemetry, "energy_session_kwh")
    txt = [f"Right now the motor is drawing **{motor:.0f} kW**" + (f" with **{regen:.0f} kW** recovering into the pack" if regen and regen < 0 else ".")]
    if trip and trip > 0:
        txt.append(f"This trip: {trip:.2f} km, {energy or 0:.2f} kWh used.")
    txt.append("Range is most sensitive to speed above 90 km/h: aero drag is dominant, and 120 vs 100 km/h costs ~30% more energy per km. Anticipate, use Eco, and let regen do the braking." + _cite(docs, "range"))
    return " ".join(txt)


def _h_winter(telemetry: dict, docs: list[dict], q: str) -> str:
    outside = _sig(telemetry, "outside_temp_c")
    bat = _sig(telemetry, "battery_temp_c")
    txt = []
    if outside is not None:
        txt.append(f"Ambient is **{outside:.0f} C** right now.")
    if outside is not None and outside < 7:
        txt.append("Winter tyres are recommended below 7 C ambient.")
    txt.append("Cold weather trims range 15-30% below -5 C (denser air, pack heating, cabin draw). Pre-condition while plugged in to move that load to grid power, set departure schedules, and keep a margin below 25% SOC before long journeys." + _cite(docs, "winter"))
    return " ".join(txt)


def _h_maintenance(telemetry: dict, docs: list[dict], q: str) -> str:
    txt = [
        "The ORION-X service interval is **24 months / 30,000 km**, whichever comes first. "
        "Key items: high-voltage system inspection, coolant, brake fluid every 3 years, cabin filter every 2 years, tyre rotation every 10,000 km, and 12 V battery check (typical life 3-5 years)."
    ]
    txt.append(_cite(docs, "maintenance"))
    if any(w.get("kind") == "12v" for w in telemetry.get("warnings", [])):
        txt.append("⚠ Your 12 V auxiliary battery flagged — get it checked; a weak 12 V battery causes sluggish accessory and telematics behaviour." + _cite(docs, "maintenance"))
    return " ".join(txt)


def _h_driving_mode(telemetry: dict, docs: list[dict], q: str) -> str:
    mode = telemetry.get("mode") or telemetry.get("driving_mode") or "eco"
    return (
        f"You're in **{mode} mode**. Eco softens torque and maximises regen (+range); "
        "Normal balances performance and 60 kW of regen; Sport lifts power limits and runs hotter, "
        "ideal for overtaking but noticeably thirstier. In Sport the range-extender may assist to keep the pack from draining." + _cite(docs, "overview")
    )


def _h_speed(telemetry: dict, docs: list[dict], q: str) -> str:
    speed = _sig(telemetry, "vehicle_speed_kph")
    accel = _sig(telemetry, "accel_mps2")
    if speed is None:
        return "I don't have a live speed reading in the grounding right now."
    if speed < 0.5:
        return "The vehicle is stationary right now."
    return f"You're travelling at **{speed:.0f} km/h**" + (f", {'accelerating' if accel and accel > 0.3 else 'steady or coasting'}." if accel is not None else ".") + " Above ~115 km/h, easing off a touch meaningfully extends range."


_HANDLERS = {
    "hello": _h_hello,
    "status": _h_status,
    "range": _h_range,
    "battery": _h_battery,
    "charging": _h_charging,
    "tire": _h_tire,
    "dtc": _h_dtc,
    "temperature": _h_temperature,
    "consumption": _h_consumption,
    "winter": _h_winter,
    "maintenance": _h_maintenance,
    "driving_mode": _h_driving_mode,
    "speed": _h_speed,
}

_DISPATCH_ORDER = (
    "hello",
    "status",
    "dtc",
    "temperature",
    "tire",
    "winter",
    "range",
    "charging",
    "battery",
    "consumption",
    "maintenance",
    "driving_mode",
    "speed",
    "general",
)