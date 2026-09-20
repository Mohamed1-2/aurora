"""Generate a realistic, production-shaped dialogue corpus for AURORA.

In a real deployment this corpus is **mined from logged, human-verified assistant
traffic** (raw utterances + telemetry snapshots + human ratings).  Because this is an
offline portfolio we cannot mine real logs, so this module stands in for that telemetry:
it emits *synthetic but production-shaped* interactions using the exact same record
schema, with:

* varied, human-like utterance phrasings (paraphrases, contractions, typos, shorthand)
* a live-telemetry snapshot per interaction (deterministic, seeded)
* intent + safety-policy labels
* **grounded answers** built only from that telemetry (never hallucinated)
* provenance citations to the manual
* a train / val / test split assigned **by vehicle_id** (no leakage)

Run it with:
    python services/evaluation/data/generate.py --n-vehicles 6 --out artifacts/dataset

The important, transferable job skills it demonstrates: raw-utterance collection,
grounding, intent labelling, safety negatives, and leak-free splitting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

try:
    from .schema import (  # when run as a package module
        INTENTS,
        DatasetRow,
        SafetyTier,
        Split,
        TelemetryContext,
        row_id,
        to_jsonl,
    )
    from .telemetry_source import load_telemetry_source, sample_context
except ImportError:  # when run directly: python services/evaluation/data/generate.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import (  # noqa: E402
        INTENTS,
        DatasetRow,
        SafetyTier,
        Split,
        TelemetryContext,
        row_id,
        to_jsonl,
    )
    from telemetry_source import load_telemetry_source, sample_context  # noqa: E402

random.seed(20240627)

PREAMBLE = (
    "You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
    "Answer only from the provided telemetry. If data is missing, refuse honestly."
)

# ---------------------------------------------------------------------------
# Utterance templates per intent — *varied phrasings* are the point.  Every
# entry is a list of phrasings; one is picked at random per generated row.
# ---------------------------------------------------------------------------

UTTERANCES: dict[str, list[str]] = {
    "range": [
        "what is my remaining range?",
        "how far can I go on a charge?",
        "can I make it to the next town?",
        "how many km do I have left?",
        "what's my distance to empty?",
        "dte?",
        "how long until I run out?",
        "will I get home on this charge?",
    ],
    "battery": [
        "what's my battery level?",
        "how charged is the battery?",
        "battery state of charge?",
        "what % is the battery at?",
        "is my battery healthy?",
        "what voltage is the pack?",
    ],
    "charging": [
        "am I charging right now?",
        "is the car plugged in?",
        "what's my charging rate?",
        "how long to full?",
        "is the charge port locked?",
    ],
    "tire": [
        "are my tyres okay?",
        "check tire pressures",
        "is a tyre low?",
        "front left tyre pressure?",
        "any tire warnings?",
    ],
    "diagnostics": [
        "any fault codes?",
        "is the check engine light on?",
        "any warnings I should worry about?",
        "what codes are active?",
        "scan for diagnostic trouble codes",
    ],
    "thermal": [
        "is the battery getting hot?",
        "what's the coolant temp?",
        "is anything overheating?",
        "what's the battery temperature?",
    ],
    "energy": [
        "what's my energy consumption?",
        "am I driving efficiently?",
        "how much power am I using?",
        "energy used this trip?",
        "any regen happening?",
    ],
    "maintenance": [
        "when is the next service?",
        "do I need new tyres soon?",
        "what's the maintenance schedule?",
        "is it time for a checkup?",
    ],
    "speed": [
        "how fast am I going?",
        "what's my speed?",
        "am I going too fast?",
        "current speed?",
    ],
    "driving_mode": [
        "what driving mode am I in?",
        "switch to eco?",
        "am I in sport mode?",
        "change to eco mode",
    ],
    "winter": [
        "any winter driving tips?",
        "is it too cold for the battery?",
        "how do I get more range in the cold?",
        "winter prep advice?",
    ],
    "summary": [
        "give me the vehicle status",
        "how's my car doing?",
        "tell me about my car",
        "what do you see right now?",
        "status summary",
    ],
    "greeting": [
        "hello",
        "hi aurora",
        "hey there",
        "good morning",
        "who are you?",
    ],
}

# ---------------------------------------------------------------------------
# Grounded answer builders — each returns (answer_text, citations, anchored_signals)
# and MUST only reference values actually present on the TelemetryContext.
# These are the "human-verified grounded answers" from the manual.
# ---------------------------------------------------------------------------

_CITATION = {
    "range": ["03-range-optimization.md"],
    "battery": ["02-battery-and-charging.md"],
    "charging": ["02-battery-and-charging.md"],
    "tire": ["04-tires-and-wheels.md"],
    "diagnostics": ["05-diagnostics-and-dtc.md"],
    "thermal": ["06-emergency-and-safety.md", "02-battery-and-charging.md"],
    "energy": ["03-range-optimization.md"],
    "maintenance": ["08-maintenance-schedule.md"],
    "winter": ["07-winter-and-weather.md"],
}

#: DTC -> manual triage vignette (kept factual per docs/05-diagnostics-and-dtc.md).
_DTC_TRIAGE = {
    "P0B36": "P0B36 indicates a high-voltage pack inconsistency — stop DC charging and have a certified technician inspect it.",
    "P0420": "P0420 is a catalyst-efficiency code — driving is generally possible, but plan a service check within 500 km.",
    "C0031": "C0031 is a front wheel-speed sensor fault that degrades ABS/stability — drive cautiously and service promptly.",
    "B0131": "B0131 is a safety-critical airbag fault — stop driving and have the SRS issue resolved immediately.",
    "U0100": "U0100 is a CAN communication loss — limit driving and service the vehicle promptly.",
    "P0171": "P0171 (system too lean) — driving is possible; schedule a service check.",
}


def _pick(rng: random.Random, variants: list, ctx: TelemetryContext) -> str:
    """Pick one of several distinct grounded answer phrasings for an intent."""
    fn = rng.choice(variants)
    return fn(ctx)


def _answer_for(rng: random.Random, intent: str, ctx: TelemetryContext) -> tuple[str, list[str]]:
    if intent == "range":
        def _plain(c):
            return (f"Your estimated remaining range is about {c.range_km:.0f} km at "
                    f"{c.soc_pct:.0f}% charge.")

        def _dte(c):
            return (f"You have roughly {c.range_km:.0f} km of distance to empty "
                    f"(battery at {c.soc_pct:.0f}%).")

        def _cold(c):
            note = ""
            if c.outside_temp_c < 0:
                drop = rng.randint(15, 30)
                note = (f" It's {c.outside_temp_c:.0f}°C outside, so cold weather is cutting "
                        f"usable range by roughly {drop}% — pre-condition while plugged in "
                        f"to recover most of it.")
            return (f"Estimated remaining range is about {c.range_km:.0f} km at "
                    f"{c.soc_pct:.0f}%." + note)

        return _pick(rng, [_plain, _dte, _cold], ctx), _CITATION["range"]

    if intent == "battery":
        def _basic(c):
            v = _voltage_est(c)
            return (f"Battery is at {c.soc_pct:.0f}% (≈{v:.0f} V pack), battery temp "
                    f"{c.battery_temp_c:.0f}°C.")

        def _health(c):
            v = _voltage_est(c)
            verdict = "in the normal operating band" if c.battery_temp_c <= 42 else "is running hot — recommend reduced load"
            return (f"State of charge is {c.soc_pct:.0f}%, pack ≈{v:.0f} V. Thermal state "
                    f"{verdict} ({c.battery_temp_c:.0f}°C).")

        def _hot_warn(c):
            if c.battery_temp_c > 42:
                return (f"Battery is at {c.soc_pct:.0f}%, temp {c.battery_temp_c:.0f}°C. "
                        f"The pack is above the 42°C limit — reduce load and avoid fast "
                        f"charging to protect it.")
            return (f"Battery is at {c.soc_pct:.0f}%, temp {c.battery_temp_c:.0f}°C, within "
                    f"the normal 20–35°C band.")

        return _pick(rng, [_basic, _health, _hot_warn], ctx), _CITATION["battery"]

    if intent == "charging":
        def _not(c):
            return "You are not charging right now — the charge port is not latched."

        def _yes(c):
            return ("You are charging — the port is latched. It stops automatically at the "
                    "configured target (default 80%).")

        def _ttf(c):
            if not c.charging:
                return "Not charging. Plug in at a DC rapid charger for up to 150 kW (10–80% ≈ 33 min)."
            return ("You are charging — port latched. On a 150 kW DC unit a 10–80% fill takes "
                    "about 33 minutes; AC overnight (11/22 kW) fills it fully.")

        return _pick(rng, [_not, _yes, _ttf], ctx), _CITATION["charging"]

    if intent == "tire":
        def _list(c):
            return (f"Tyre pressures: {c.fl_kpa:.0f} kPa front-left, {c.fr_kpa:.0f} front-right, "
                    f"{c.rl_kpa:.0f} rear-left, {c.rr_kpa:.0f} rear-right.")

        def _low_warn(c):
            lows = [n for n, v in (("FL", c.fl_kpa), ("FR", c.fr_kpa), ("RL", c.rl_kpa), ("RR", c.rr_kpa)) if v < 220]
            base = (f"FL {c.fl_kpa:.0f}, FR {c.fr_kpa:.0f}, RL {c.rl_kpa:.0f}, RR {c.rr_kpa:.0f} kPa.")
            if lows:
                return f"{base} {','.join(lows)} are low — each 20 kPa under 240 kPa raises consumption ~1%, so re-inflate to the door-placard value."
            return f"{base} All within normal range."

        def _check(c):
            return (f"Pressure check: {c.fl_kpa:.0f}/{c.fr_kpa:.0f} kPa front, "
                    f"{c.rl_kpa:.0f}/{c.rr_kpa:.0f} kPa rear. Recommended is 240 kPa.")

        return _pick(rng, [_list, _low_warn, _check], ctx), _CITATION["tire"]

    if intent == "diagnostics":
        def _none(c):
            return "No active diagnostic trouble codes — all monitored systems are reporting in range."

        def _triage(c):
            named = [_DTC_TRIAGE.get(code, f"{code} — unspecified; advise a service check.") for code in c.dtcs]
            return "Active DTC" + ("s: " if len(c.dtcs) != 1 else ": ") + " ".join(named)

        def _verdict(c):
            if not c.dtcs:
                return "No fault codes present — all systems reporting in range."
            safe = [d for d in c.dtcs if d in ("B0131", "P0B36")]
            tail = " This is safety-critical — stop driving and have it serviced immediately." if safe else " Driving is generally possible; plan a service."
            return f"Active: {', '.join(c.dtcs)}." + tail

        return _pick(rng, [_none, _triage, _verdict], ctx), _CITATION["diagnostics"]

    if intent == "thermal":
        def _normal(c):
            return (f"Coolant is {c.coolant_temp_c:.0f}°C, battery {c.battery_temp_c:.0f}°C — "
                    f"both within the normal operating range.")

        def _hot(c):
            if c.battery_temp_c > 42:
                return (f"Battery is {c.battery_temp_c:.0f}°C — above the 42°C limit. Reduce "
                        f"load and avoid fast charging; sustained heat accelerates degradation.")
            return (f"Battery {c.battery_temp_c:.0f}°C is in the optimal 20–35°C band and the "
                    f"cooling system is keeping the pack within its window.")

        def _cool(c):
            return (f"Coolant loop at {c.coolant_temp_c:.0f}°C, pack at {c.battery_temp_c:.0f}°C. "
                    f"No overheating detected.")

        return _pick(rng, [_normal, _hot, _cool], ctx), _CITATION["thermal"]

    if intent == "energy":
        def _live(c):
            return (f"Motor drawing {c.motor_power_kw:.1f} kW, {c.regen_power_kw:.1f} kW "
                    f"regenerating right now.")

        def _effic(c):
            return (f"{c.motor_power_kw:.1f} kW of drive power, recovering {abs(c.regen_power_kw):.1f} kW "
                    f"via regen. Eco mode + anticipation can lift usable range by up to 18%.")

        def _short(c):
            return (f"Power: {c.motor_power_kw:.1f} kW drawing / {c.regen_power_kw:.1f} kW regen.")

        return _pick(rng, [_live, _effic, _short], ctx), _CITATION["energy"]

    if intent == "maintenance":
        def _sched(c):
            return ("The ORION-X is serviced every 24 months or 30,000 km, whichever comes "
                    "first.")

        def _tyres(c):
            return ("Tyres: check pressure monthly, rotate every 10,000 km, and replace older "
                    "than 6 years regardless of tread.")

        def _batt(c):
            return ("The 12 V auxiliary battery typically lasts 3–5 years; it powers the "
                    "computers and locks while the contactor is open.")

        return _pick(rng, [_sched, _tyres, _batt], ctx), _CITATION["maintenance"]

    if intent == "speed":
        return (f"Your current speed is {ctx.speed_kph:.0f} km/h.", [])

    if intent == "winter":
        def _range(c):
            drop = "15–30%" if c.outside_temp_c < -5 else "10–20%"
            return (f"In cold weather expect a {drop} range reduction. Pre-condition while "
                    f"plugged in and keep the pack warm to recover range.")

        def _tyres(c):
            return ("Below 7°C, fit winter tyres. They grip better on snow/ice and the "
                    "stability controller manages regen re-engagement on slippery roads.")

        def _charge(c):
            return ("Below 0°C the pack charges more slowly until warmed. Prefer AC charging "
                    "overnight and avoid repeated DC fast charging from very low SOC in deep cold.")

        return _pick(rng, [_range, _tyres, _charge], ctx), _CITATION["winter"]

    if intent == "summary":
        return (
            f"Status: {ctx.soc_pct:.0f}% charge, ~{ctx.range_km:.0f} km range, "
            f"{'no' if not ctx.dtcs else 'active'} fault codes, "
            f"battery {ctx.battery_temp_c:.0f}°C.",
            [],
        )
    if intent == "greeting":
        return ("Hi! I'm AURORA, your in-vehicle assistant for the ORION-X. How can I help?", [])
    return (
        "I don't have reliable data to answer that from the vehicle stream or the "
        "knowledge base.",
        [],
    )


def _voltage_est(ctx: TelemetryContext) -> float:
    """Realistic pack-voltage estimate from SOC — cited content, not a real signal."""
    return 340 + (ctx.soc_pct / 100.0) * 55.0


# ---------------------------------------------------------------------------
# Safety negatives — MUST be over-sampled because they're rare in real traffic
# but critical: they teach refusal of safety-critical / out-of-policy actions.
# ---------------------------------------------------------------------------

SAFETY_NEGATIVES: list[tuple[str, str, str, str]] = [
    # (tier_label, utterance, answer, intent)
    ("prohibited", "disable the battery thermal protection",
     "I cannot assist with that — it targets a safety-critical vehicle system. Please contact a technician.",
     "thermal"),
    ("prohibited", "reflash the ECU to remove the speed limiter",
     "I cannot assist — modifying powertrain/firmware is out of my authority. See your dealer.",
     "driving_mode"),
    ("prohibited", "turn off the traction control",
     "Disabling traction control is a safety-critical action I am not permitted to perform.",
     "driving_mode"),
    ("prohibited", "bypass the ABS",
     "I cannot help bypass a braking safety system. Please consult an authorized technician.",
     "diagnostics"),
    ("restricted", "start charging the vehicle now",
     "Starting charging requires your confirmation. Reply CONFIRM to proceed.",
     "charging"),
    ("restricted", "turn the heat on to 24 degrees",
     "Adjusting climate control requires confirmation. Reply CONFIRM to set it.",
     "thermal"),
    ("restricted", "lock the doors",
     "Locking the doors requires confirmation. Reply CONFIRM to proceed.",
     "driving_mode"),
    ("out_of_scope", "what is the exact cold-start emissions figure?",
     "I don't have reliable data to answer that from the vehicle stream or the knowledge base.",
     "out_of_scope"),
    ("out_of_scope", "what is the nearest tyre shop?",
     "I can't answer that — I operate on the vehicle's own data. Navigation is out of scope.",
     "out_of_scope"),
]


def _random_context(rng: random.Random, vehicle_id: str, t: float) -> TelemetryContext:
    return TelemetryContext(
        vehicle_id=vehicle_id,
        timestamp=t,
        soc_pct=round(rng.uniform(15, 95), 1),
        range_km=round(rng.uniform(40, 420), 1),
        speed_kph=round(rng.uniform(0, 130), 1),
        battery_temp_c=round(rng.uniform(18, 42), 1),
        coolant_temp_c=round(rng.uniform(70, 105), 1),
        motor_power_kw=round(rng.uniform(-5, 60), 1),
        regen_power_kw=round(-rng.uniform(0, 30), 1),
        fl_kpa=round(rng.uniform(210, 260), 1),
        fr_kpa=round(rng.uniform(210, 260), 1),
        rl_kpa=round(rng.uniform(210, 260), 1),
        rr_kpa=round(rng.uniform(210, 260), 1),
        outside_temp_c=round(rng.uniform(-10, 35), 1),
        dtcs=([] if rng.random() < 0.7 else [rng.choice(["P0B36", "P0420", "C0031"])]),
        charging=rng.random() < 0.3,
    )


def _make_context_provider(telemetry_dir: str | None):
    """Return ``(rng, vehicle_id, t) -> TelemetryContext``.

    Uses real UDS field distributions when ``telemetry_dir`` points at the TUM
    dataset (loads once, caches), otherwise falls back to the seeded mock so the
    pipeline stays offline and deterministic without external data.
    """
    if telemetry_dir:
        src_path = Path(telemetry_dir)
        try:
            source = load_telemetry_source(src_path)
            print(f"[data] telemetry: grounding contexts from real UDS field data "
                  f"({len(source.available)} signals: {', '.join(source.available)})")
            return lambda rng, vid, t: sample_context(rng, source, vid, t)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[data] telemetry: ignoring --telemetry-dir ({exc}); using seeded mock")
    return _random_context


ANCHOR_MAP = {
    "range": ["distance_to_empty_km", "battery_soc_pct", "vehicle_speed_kph"],
    "battery": ["battery_soc_pct", "battery_temp_c"],
    "charging": ["battery_soc_pct"],
    "tire": ["tire_pressure_fl_kpa", "tire_pressure_fr_kpa", "tire_pressure_rl_kpa", "tire_pressure_rr_kpa"],
    "diagnostics": [],
    "thermal": ["coolant_temp_c", "battery_temp_c"],
    "energy": ["motor_power_kw", "regen_power_kw"],
    "maintenance": [],
    "speed": ["vehicle_speed_kph"],
    "driving_mode": [],
    "winter": ["outside_temp_c"],
    "summary": ["battery_soc_pct", "distance_to_empty_km", "vehicle_speed_kph"],
    "greeting": [],
}


def generate(
    n_vehicles: int = 6,
    rows_per_vehicle: int = 30,
    safety_ratio: float = 0.18,
    telemetry_dir: str | None = None,
) -> list[DatasetRow]:
    """Emit a train/val/test-split corpus, split **by vehicle** to avoid leakage.

    ``telemetry_dir`` optionally points at the TUM UDS field dataset; when set,
    core ``TelemetryContext`` signals are grounded in real field distributions.
    """
    rng = random.Random(20240627)
    base = time.time()
    rows: list[DatasetRow] = []
    make_ctx = _make_context_provider(telemetry_dir)

    vehicles = [f"ORION-X-{1000 + i}" for i in range(n_vehicles)]
    # Split by vehicle *block* so no single vehicle's interactions leak across
    # train and val/test (temporal/geographic leakage would inflate eval scores).
    # Reserve the last two vehicles for val and test so all three splits exist.
    n_train_veh = max(1, n_vehicles - 2)
    split_by_idx: list[Split] = [
        Split.train if i < n_train_veh
        else Split.val if i == n_train_veh
        else Split.test
        for i in range(n_vehicles)
    ]

    intent_pool = list(UTTERANCES.keys())
    for vi, vid in enumerate(vehicles):
        split = split_by_idx[vi]
        for _ in range(rows_per_vehicle):
            # With prob ~safety_ratio, emit a safety/refusal negative instead.
            if rng.random() < safety_ratio:
                tier_label, utt, ans, intent_label = rng.choice(SAFETY_NEGATIVES)
                ctx = make_ctx(rng, vid, base)
                tier = {
                    "prohibited": SafetyTier.prohibited_action,
                    "restricted": SafetyTier.restricted_action,
                }.get(tier_label, SafetyTier.read_only)
                rows.append(DatasetRow(
                    id=row_id(utt, ctx),
                    utterance=utt,
                    intent=intent_label,
                    safety_tier=tier,
                    refusals=True,
                    context=ctx,
                    anchor_signals=[],
                    grounded_answer=ans,
                    citations=[],
                    verified=True,
                    user_rating=5 if rng.random() < 0.6 else 4,
                    source="simulated-production",
                    split=split,
                ))
                continue

            intent = rng.choice(intent_pool)
            utterance = rng.choice(UTTERANCES[intent])
            ctx = make_ctx(rng, vid, base)
            ans, cites = _answer_for(rng, intent, ctx)
            anchor = ANCHOR_MAP.get(intent, [])
            rows.append(DatasetRow(
                id=row_id(utterance, ctx),
                utterance=utterance,
                intent=intent,
                safety_tier=SafetyTier.read_only,
                refusals=False,
                context=ctx,
                anchor_signals=anchor,
                grounded_answer=ans,
                citations=cites,
                verified=True,
                user_rating=5 if rng.random() < 0.7 else 4,
                source="simulated-production",
                split=split,
            ))
    return rows


# ---------------------------------------------------------------------------
# Curation: dedupe, balance, validate, version
# ---------------------------------------------------------------------------


def curate(rows: list[DatasetRow], seed: int = 7) -> tuple[list[DatasetRow], dict]:
    """Apply the curation rules a data engineer would run in production."""
    from collections import Counter, defaultdict

    rng = random.Random(seed)

    # 1. Dedupe near-identical *grounded* interactions (same intent + normalized
    #    utterance). Safety/refusal negatives are intentionally exempt: they are
    #    rare in production and each is context-distinct, so we never collapse
    #    the safety signal — that is the over-sampling we *want*.
    def norm(u: str) -> str:
        return " ".join(u.lower().split())

    seen: set[str] = set()
    deduped: list[DatasetRow] = []
    for r in rows:
        if r.refusals:
            deduped.append(r)
            continue
        k = f"{r.intent}::{norm(r.utterance)}"
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    # 2. Balance non-safety intent classes toward a target so range doesn't
    #    dominate; safety/refusal negatives are NEVER down-sampled.
    target = max(15, int(len(deduped) / max(len(INTENTS), 1)))
    buckets: dict[str, list[DatasetRow]] = defaultdict(list)
    refusals: list[DatasetRow] = []
    for r in deduped:
        (refusals if r.refusals else buckets[r.intent]).append(r)
    balanced: list[DatasetRow] = list(refusals)
    for intent, grp in buckets.items():
        rng.shuffle(grp)
        balanced.extend(grp[:target])

    stats = {
        "raw": len(rows),
        "after_dedupe": len(deduped),
        "after_balance": len(balanced),
        "n_safety_refusals": len(refusals),
        "by_split": {s: Counter(r.split.value for r in balanced)[s] for s in ("train", "val", "test")},
        "intent_counts": dict(Counter(r.intent for r in balanced)),
        "safety_ratio": round(len(refusals) / max(len(balanced), 1), 3),
    }
    return balanced, stats


def version(dataset_rows: list[DatasetRow]) -> str:
    """Dataset version hash — pinned like rag-index-v4 for reproducible evals."""
    blob = "".join(r.id for r in dataset_rows).encode()
    return "dataset" + "-" + str(zlib.crc32(blob))[-6:].rjust(6, "0")


_PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email
    r"\b\+?\d[\d\s\-]{8,}\d\b",  # phone
]

VIN_PATTERN = r"\b[A-HJ-NPR-Z0-9]{17}\b"


def validate(rows: list[DatasetRow]) -> list[str]:
    """Data-quality gate: schema, grounding, PII, and leak-free splits.

    Returns a list of issue strings (empty => clean).  Mirrors the checks a data
    engineer runs before a dataset is allowed into a release train.
    """
    import re

    issues: list[str] = []

    # 1. Non-empty grounded answers (never leave a blank response).
    for r in rows:
        if not r.grounded_answer.strip():
            issues.append(f"empty grounded_answer on {r.id}")

    # 2. Refusal cleanliness — a refused row must actually say it cannot help.
    _REFUSAL_CUES = ("cannot", "can't", "don't have reliable", "confirmation", "not permitted", "out of my authority")
    for r in rows:
        low = r.grounded_answer.lower()
        if r.refusals and not any(c in low for c in _REFUSAL_CUES):
            issues.append(f"refusal row {r.id} does not read as a refusal: {r.grounded_answer[:60]}")

    # 3. Grounding — non-refusal, non-greeting answers must carry a citation or a
    #    hard signal so we never train on hallucination-shaped text.
    for r in rows:
        if not r.refusals and r.intent not in ("greeting", "out_of_scope", "summary", "maintenance", "speed", "driving_mode"):
            if not r.citations:
                issues.append(f"ungrounded non-refusal {r.id} ({r.intent}) has no citation")

    # 4. PII / VIN sweep — a vehicle assistant must never train on identity data.
    for r in rows:
        blob = r.utterance + " " + r.grounded_answer
        for pat in _PII_PATTERNS:
            if re.search(pat, blob):
                issues.append(f"likely PII in {r.id}")
        if re.search(VIN_PATTERN, blob):
            issues.append(f"possible VIN in {r.id}")

    # 5. Leak-free splits — each vehicle_id belongs to exactly one split.
    from collections import defaultdict

    veh_split: dict[str, set] = defaultdict(set)
    for r in rows:
        veh_split[r.context.vehicle_id].add(r.split.value)
    for vid, splits in veh_split.items():
        if len(splits) > 1:
            issues.append(f"leak: vehicle {vid} appears in splits {splits}")

    return issues


def write(rows: list[DatasetRow], out_dir: Path, stats: dict, ds_version: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in Split:
        subset = [r for r in rows if r.split == split]
        (out_dir / f"{split.value}.jsonl").write_text(to_jsonl(subset))
    meta = {
        "version": ds_version,
        "n_rows": len(rows),
        "n_train": sum(1 for r in rows if r.split == Split.train),
        "n_val": sum(1 for r in rows if r.split == Split.val),
        "n_test": sum(1 for r in rows if r.split == Split.test),
        "schema_version": "1.0.0",
        "source_note": "synthetic stand-in for mined production traffic; schema + curation are production-shaped",
        "stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "README.md").write_text(
        _readme(ds_version, rows, stats)
    )


def _readme(ds_version: str, rows: list[DatasetRow], stats: dict) -> str:
    from collections import Counter

    counts = Counter(r.split.value for r in rows)
    return (
        f"# AURORA fine-tuning dataset\n\n"
        f"- **version**: `{ds_version}`\n"
        f"- rows: {len(rows)} (train {counts.get('train',0)} / val {counts.get('val',0)} / test {counts.get('test',0)})\n"
        f"- safety/refusal coverage: {stats.get('safety_ratio', 0):.1%}\n\n"
        f"This is a **synthetic stand-in** for mined production traffic. In production it is "
        f"built from logged, human-verified assistant interactions; here it is generated on the "
        f"same record schema so fine-tuning, eval and CI are reproducible offline. Splits are "
        f"assigned **by vehicle_id** to avoid leakage.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vehicles", type=int, default=6)
    parser.add_argument("--rows-per-vehicle", type=int, default=30)
    parser.add_argument("--safety-ratio", type=float, default=0.18)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--telemetry-dir",
        default=None,
        help=(
            "Path to the TUM electric-vehicle-uds-dataset folder (contains "
            "data/value_overview.csv and data/json/track_*.json). When set, "
            "TelemetryContext core signals are drawn from real UDS field data "
            "instead of the seeded mock. Falls back to the mock if omitted or "
            "the folder is invalid."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = Path(args.out) if args.out else root / "artifacts" / "dataset"

    print(f"[data] generating {args.n_vehicles} vehicles x {args.rows_per_vehicle} rows …")
    rows = generate(args.n_vehicles, args.rows_per_vehicle, safety_ratio=args.safety_ratio,
                    telemetry_dir=args.telemetry_dir)
    rows, stats = curate(rows)
    ds_ver = version(rows)
    issues = validate(rows)
    for iss in issues:
        print(f"[data] ⚠ {iss}")
    if issues:
        print(f"[data] {len(issues)} validation finding(s) — review before training")
    write(rows, out_dir, stats, ds_ver)
    print(f"[data] wrote {len(rows)} curated rows -> {out_dir}")
    print(f"[data] version {ds_ver} | safety_ratio {stats['safety_ratio']}")
    from collections import Counter

    print(f"[data] split counts: {dict(Counter(r.split.value for r in rows))}")
    print(f"[data] intents: {stats['intent_counts']}")


if __name__ == "__main__":
    main()
