"""WS4 proactive-detection golden evaluation and deploy gate (deterministic).

Usage:
    python scripts/eval_proactive.py [--no-gate]

Replays scripted 10 Hz frame timelines through the pure ``ProactiveEvaluator``
with an injected simulated clock, so data-availability gating, confirmation,
cooldown suppression, monotonic escalation and resolution are all reproducible
instead of wall-clock dependent. Reports per-scenario alert logs plus
aggregate precision / recall / false-alert metrics, writes
``artifacts/proactive/eval/report.json`` and exits non-zero on a missed gate.

Gate:
    precision == 1.0, recall == 1.0, zero false-alert runs,
    all cooldown/escalation/fallback assertions hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services/assistant-api"))
sys.path.insert(0, str(REPO / "shared"))

from app.core.proactive_engine import ProactiveEvaluator  # noqa: E402
from vehicle_ai.schemas import (  # noqa: E402
    CalibratedSoCDropResponse,
    PredictionExplanation,
)

GOLDEN = REPO / "services/evaluation/proactive/golden.yml"
HZ = 10.0
T0 = 1_788_800_000.0

BASE_SIGNALS = {
    "vehicle_speed_kph": 70.0,
    "battery_soc_pct": 60.0,
    "outside_temp_c": 12.0,
    "coolant_temp_c": 85.0,
    "battery_voltage_v": 350.0,
    "hvac_power_kw": 1.2,
    "distance_to_empty_km": 180.0,
    "battery_temp_c": 32.0,
    "charge_rate_kw": 0.0,
    "tire_pressure_fl_kpa": 240.0,
    "tire_pressure_fr_kpa": 240.0,
    "tire_pressure_rl_kpa": 240.0,
    "tire_pressure_rr_kpa": 240.0,
}

GATE = {"precision": 1.0, "recall": 1.0, "false_alerts": 0}


def _to_pred(drop: float, status: str):
    return CalibratedSoCDropResponse(
        physics_soc_drop_pct=drop,
        ml_residual_pct=0.0,
        calibrated_soc_drop_pct=drop,
        speed_band="med",
        lambda_applied=1.0,
        status=status,
        model_version="lgbm-socdrop-v7" if status == "active" else None,
        fallback_reason="" if status == "active" else "ML model unavailable",
    )


EXPLAIN_ACTIVE = PredictionExplanation(
    status="active", model_version="lgbm-socdrop-v7", to_prediction=_to_pred(10.0, "active")
)
EXPLAIN_FALLBACK = PredictionExplanation(
    status="physics_only_fallback",
    fallback_reason="model unavailable",
    to_prediction=_to_pred(10.0, "physics_only_fallback"),
)


def build_timeline(profile: str, duration_s: float, intensity: float) -> list[dict]:
    """Generate the scripted 10 Hz frame timeline for a scenario profile."""
    n = int(duration_s * HZ)
    dt = 1.0 / HZ
    frames: list[dict] = []

    def frame(t: float, overrides: dict, dtcs: list | None = None) -> dict:
        sig = dict(BASE_SIGNALS)
        sig.update(overrides)
        return {"source": "eval", "vehicle_id": "ORION-X-TEST-01", "ts": round(t, 3),
                "signals": sig, "active_dtcs": dtcs if dtcs is not None else []}

    for i in range(n + 1):
        t = T0 + i * dt
        elapsed = i * dt
        frac = elapsed / duration_s if duration_s else 0.0
        overrides: dict = {}
        if profile == "leak":
            # Sustained 0.25 kPa/s leak (15 kPa/30s -> -15 kPa/min slope) so the
            # detector keeps firing to the end; cooldown (900 s) is what caps it.
            overrides = {"tire_pressure_rr_kpa": max(90.0, 240.0 - 0.25 * elapsed)}
        elif profile == "battery_heat":
            overrides = {"battery_temp_c": 32.0 + 30.0 * intensity * frac}
        elif profile == "range":
            overrides = {"distance_to_empty_km": 30.0}
        elif profile == "charge":
            overrides = {"charge_rate_kw": 11.0}
        elif profile == "charge_stalled":
            overrides = {"charge_rate_kw": 0.02}
        elif profile in ("discharge", "discharge_fallback"):
            overrides = {"battery_soc_pct": 90.0 - 12.0 * intensity * frac}

        dtcs = None
        if profile == "dtc" and elapsed >= 15.0:
            dtcs = [{"code": "C0031", "severity": "critical",
                     "description": "braking system — pump circuit"}]
        frames.append(frame(t, overrides, dtcs))
    return frames


class ScenarioTelemetry:
    """Segment of a timeline with the interface the engine reads."""

    def __init__(self, timeline: list[dict], charging: bool, signals_extra: set[str]):
        self._timeline = timeline
        self._charging = charging
        self._signals = set(BASE_SIGNALS) | signals_extra
        self._t = T0

    def set_t(self, ts: float) -> None:
        self._t = ts

    @property
    def frames(self) -> list[dict]:
        cutoff = self._t + (10.0 / HZ)
        return [f for f in self._timeline if f["ts"] <= cutoff]

    def has_signal(self, sig: str) -> bool:
        return sig in self._signals

    def signal_series(self, sig: str, max_age_s: float = 45.0, stride_s: float = 1.0):
        pts = []
        for f in self._timeline:
            if f["ts"] > self._t:
                break
            if f["ts"] < self._t - max_age_s:
                continue
            v = f["signals"].get(sig)
            if v is None:
                continue
            pts.append((f["ts"], v))
        if stride_s and stride_s > 0.0:
            step = max(1, int(round(stride_s * HZ)))
            pts = pts[::step]
        return pts

    def grounding(self) -> dict:
        latest = self.frames[-1] if self.frames else {}
        return {
            "vehicle_id": latest.get("vehicle_id", "ORION-X-TEST-01"),
            "signals": dict(latest.get("signals", {})),
            "charging": self._charging,
            "dtcs": [d for d in latest.get("active_dtcs", [])],
        }


SCENARIO_META = {
    "none":             {"charging": False, "extra": set(), "explain": None},
    "leak":             {"charging": False, "extra": set(), "explain": None},
    "battery_heat":     {"charging": False, "extra": set(), "explain": None},
    "range":            {"charging": False, "extra": set(), "explain": None},
    "dtc":              {"charging": False, "extra": set(), "explain": None},
    "charge":           {"charging": True,  "extra": {"charge_rate_kw"}, "explain": None},
    "charge_stalled":   {"charging": True,  "extra": {"charge_rate_kw"}, "explain": None},
    "discharge":        {"charging": False, "extra": set(), "explain": lambda: EXPLAIN_ACTIVE},
    "discharge_fallback": {"charging": False, "extra": set(), "explain": lambda: EXPLAIN_FALLBACK},
}


def run_scenario(scenario: dict, poll_interval_s: float) -> dict:
    s_id = scenario["id"]
    profile = scenario["profile"]
    meta = SCENARIO_META[profile]
    duration = float(scenario["duration_s"])
    timeline = build_timeline(profile, duration, float(scenario["intensity"]))
    tele = ScenarioTelemetry(
        timeline, charging=meta["charging"], signals_extra=meta["extra"]
    )
    simulated = {"t": T0}
    ev = ProactiveEvaluator(
        tele,
        get_explanation=meta["explain"],
        clock=lambda: simulated["t"],
    )
    polls = max(1, int(duration // poll_interval_s))

    emissions: list[dict] = []
    prev: dict[str, tuple[int, str]] = {}
    for poll_i in range(polls + 1):
        elapsed = poll_i * poll_interval_s
        simulated["t"] = T0 + elapsed
        tele.set_t(simulated["t"])
        snap = asyncio.run(ev.poll())
        merged: dict[str, tuple[int, str]] = {}
        for e in list(snap.active) + list(snap.recent):
            cur = prev.get(e.event_id, (0, ""))
            merged[e.event_id] = (max(cur[0], e.occurrence_count), e.severity.value)
        for eid, (occ, sev) in merged.items():
            old = prev.get(eid, (0, ""))
            if occ > old[0]:
                emissions.append({
                    "t_s": round(elapsed, 1),
                    "event_id": eid,
                    "type": next((e.type for e in list(snap.active) + list(snap.recent) if e.event_id == eid), "?"),
                    "severity": sev,
                    "occurrence": occ,
                    "status": next((e.status for e in list(snap.active) + list(snap.recent) if e.event_id == eid), "?"),
                })
        prev = merged

    final = ev.snapshot(simulated["t"])
    stats = dict(ev.stats)
    return {"id": s_id, "emissions": emissions, "stats": stats, "active": [e.type for e in final.active],
            "recent": [(e.type, e.severity.value, e.occurrence_count, e.status) for e in final.recent]}


def cooldown_checks(scenarios: dict[str, dict]) -> dict[str, dict]:
    """Assert cooldown/repeat/escalation/fallback semantics on the run logs."""
    checks: dict[str, dict] = {}

    leak = scenarios["leak"]["emissions"]
    checks["leak_suppressed_inside_cooldown"] = {
        "ok": len(leak) == 1, "detail": f"{len(leak)} emission(s) in 600 s (cooldown 900 s)"}

    rng = scenarios["range"]["emissions"]
    cooldown_ok = False
    detail = ""
    if len(rng) >= 2:
        first = rng[0]["t_s"]
        repeat_after_cooldown = any(e["t_s"] - first >= 600.0 for e in rng[1:])
        cooldown_ok = repeat_after_cooldown and rng[-1]["occurrence"] == 2 and rng[0]["occurrence"] == 1
        detail = f"first@{first}s repeat@{rng[1]['t_s']}s occ={rng[-1]['occurrence']} repeat_after_600s={repeat_after_cooldown}"
    elif len(rng) == 1:
        detail = f"single emission, no cooldown repeat in {scenarios['range']['emissions'][0]['t_s']}s window"
    checks["low_range_cooldown_repeat"] = {"ok": cooldown_ok, "detail": detail}

    bat = scenarios["battery_heat"]["emissions"]
    sev_seq = [e["severity"] for e in bat]
    monotonic = sev_seq == ["warning", "critical"] or sev_seq == ["warning"]
    checks["battery_monotonic_escalation"] = {
        "ok": "critical" in sev_seq and "warning" in sev_seq and monotonic,
        "detail": f"severity sequence {sev_seq}", "emissions": len(bat)}

    dtc = scenarios["dtc"]["emissions"]
    checks["dtc_persistent_no_repeat_in_cooldown"] = {
        "ok": len(dtc) == 1, "detail": f"{len(dtc)} emission(s) over 600 s"}

    dis = scenarios["discharge"]["emissions"]
    coverage_ok = bool(dis) and dis[0]["t_s"] >= 300.0 and dis[0]["status"] == "full"
    checks["discharge_coverage_gate"] = {"ok": coverage_ok,
        "detail": f"first@{dis[0]['t_s']}s status={dis[0]['status']}" if dis else "no emission"}

    fb = scenarios["discharge_fallback"]["emissions"]
    fb_event = fb[0] if fb else {}
    fallback_ok = bool(fb) and fb_event.get("status") == "fallback" and fb_event.get("severity") == "info"
    checks["discharge_fallback_not_full"] = {"ok": fallback_ok,
        "detail": f"status={fb_event.get('status')} severity={fb_event.get('severity')}" if fb else "no emission"}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    doc = yaml.safe_load(Path(args.golden).read_text(encoding="utf-8"))
    poll_interval_s = float(doc["poll_interval_s"])

    runs = [run_scenario(s, poll_interval_s) for s in doc["scenarios"]]
    by_id = {run["id"]: run for run in runs}
    expectations = {s["id"]: s.get("expected", []) for s in doc["scenarios"]}

    print(f"\n{'scenario':<18} {'emissions (type@severity, t_s, occ, status)':<82} active")
    for run in runs:
        logs = ", ".join(
            f"{e['type']}@{e['severity']}(t={e['t_s']:.0f},occ={e['occurrence']},{e['status']})"
            for e in run["emissions"]
        ) or "—"
        print(f"{run['id']:<18} {logs:<82} {run['active']}")

    checks = cooldown_checks(by_id)
    print("\ncooldown / escalation / fallback checks:")
    for name, c in checks.items():
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")

    tp, fp, expected_total = 0, 0, 0
    fp_runs: list[str] = []
    missed: list[str] = []
    for run in runs:
        expected = expectations[run["id"]]
        expected_types = {e["type"] for e in expected}
        expected_total += len(expected)
        emitted: dict[str, set[str]] = {}
        for e in run["emissions"]:
            emitted.setdefault(e["type"], set()).add(e["severity"])
        run_tp = sum(1 for e in expected if e["type"] in emitted and e["severity"] in emitted[e["type"]])
        run_fp = sum(1 for typ, sevs in emitted.items()
                     if typ not in expected_types or not any(
                         e["type"] == typ and e["severity"] in sevs for e in expected))
        tp += run_tp
        fp += run_fp
        if run_fp:
            fp_runs.append(run["id"])
        for e in expected:
            if not (e["type"] in emitted and e["severity"] in emitted[e["type"]]):
                missed.append(f"{run['id']}:{e['type']}@{e['severity']}")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / expected_total if expected_total else 1.0
    agg = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "true_positives": tp,
        "false_positives": fp,
        "expected_events": expected_total,
        "false_alert_runs": fp_runs,
        "missed": missed,
        "cooldown_checks": checks,
    }
    print("\naggregate:")
    for k in ("precision", "recall", "true_positives", "false_positives", "expected_events"):
        print(f"  {k:16s} {agg[k]}")
    print(f"  false_alert_runs : {fp_runs}")
    print(f"  missed           : {missed}")

    report = {
        "golden": args.golden,
        "poll_interval_s": poll_interval_s,
        "runs": {run["id"]: {"emissions": run["emissions"], "stats": run["stats"]} for run in runs},
        "metrics": {k: agg[k] for k in ("precision", "recall", "true_positives", "false_positives", "expected_events")},
        "false_alert_runs": fp_runs,
        "missed": missed,
        "cooldown_checks": checks,
        "gate": GATE,
    }
    out_path = REPO / "artifacts/proactive/eval/report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}")

    if args.no_gate:
        return 0
    failures = []
    if precision < GATE["precision"]:
        failures.append("precision")
    if recall < GATE["recall"]:
        failures.append("recall")
    if fp > GATE["false_alerts"]:
        failures.append("false_alerts")
    failures += [name for name, c in checks.items() if not c["ok"]]
    if failures:
        print("GATE FAILED:", ", ".join(failures))
        return 1
    print("GATE PASSED: perfect precision/recall, zero false alerts, all lifecycle checks hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())