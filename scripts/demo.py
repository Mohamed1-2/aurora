"""AURORA — end-to-end platform demo.

Walks every layer of the stack: telemetry stream -> RAG -> LLM gateway routing
-> assistant grounding -> drift monitoring -> eval gate. Assumes the stack is
already running (locally via `make services` or via `docker compose up -d`).

Run:  .venv/bin/python scripts/demo.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import httpx

BASE = os.environ.get("AURORA_DEMO_BASE", "")
ASSISTANT = os.environ.get("AURORA_ASSISTANT_URL", BASE + "http://localhost:8100").replace(BASE + "http", "http")
TELEMETRY = f"{BASE or 'http://localhost'}:8101"
GATEWAY = f"{BASE or 'http://localhost'}:8102"
RAG = f"{BASE or 'http://localhost'}:8103"
MONITOR = f"{BASE or 'http://localhost'}:8104"

C = itertools.cycle(["cyan", "green", "yellow", "magenta"])
ALWAYS = ["cyan", "green", "yellow", "magenta", "blue", "red", "white"]


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}")


def _ascii_ok(text: str, ok: bool) -> str:
    return f"[{'OK ' if ok else 'FAIL'}] {text}"


def _client() -> httpx.Client:
    return httpx.Client(timeout=10.0)


def check_health(port: int, name: str) -> None:
    with _client() as c:
        try:
            r = c.get(f"{BASE or 'http://localhost'}:{port}/healthz")
            print(_ascii_ok(f"{name:<22} :{port} ({r.json().get('status', '?')})", r.status_code == 200))
        except Exception as e:  # noqa: BLE001
            print(_ascii_ok(f"{name:<22} :{port} (down)", False))


def demo_telemetry() -> None:
    _banner("1 · TELEMETRY SIMULATOR  (physics-lite ORION-X stream)")
    with _client() as c:
        latest = c.get(f"{TELEMETRY}/latest").json()
        window = c.get(f"{TELEMETRY}/window?seconds=10").json()
        key = {
            "battery_soc_pct": "SOC",
            "distance_to_empty_km": "Range",
            "vehicle_speed_kph": "Speed",
            "motor_power_kw": "Motor",
            "regen_power_kw": "Regen",
            "coolant_temp_c": "Coolant",
            "battery_temp_c": "Battery temp",
        }
        print(f"  vehicle: {latest['vehicle_id']}  seq #{latest['seq']}  scenario: {latest['driving_mode']}")
        vals = {k: latest["signals"][k] for k in key if k in latest["signals"]}
        print("  live:", "  ".join(f"{short}={vals[k]:.0f}" for k, short in key.items() if k in vals))
        w = window.get("mean", {})
        print(f"  window: {window.get('frames', '-')} frames / {window.get('window_seconds', '-')}s mean SOC {w.get('battery_soc_pct', 0):.1f}%  "
              f"mean motor {w.get('motor_power_kw', 0):.1f} kW")


def demo_rag() -> None:
    _banner("2 · RAG RETRIEVAL  (BM25 over the ORION-X manual)")
    with _client() as c:
        for q in ("low tyre pressure warning", "diagnostic trouble code p0420"):
            r = c.post(f"{RAG}/retrieve", json={"query": q, "top_k": 2}).json()
            print(f"  query: {q!r}")
            for d in r.get("top_k", [])[:2]:
                print(f"    -> {d['source']:<44} bm25={d['score']:.2f}")


def demo_gateway() -> None:
    _banner("3 · LLM GATEWAY  (routing · circuit breaker · cache · cost)")
    with _client() as c:
        provs = c.get(f"{GATEWAY}/providers").json()
        for name, info in provs.items():
            print(f"  provider: {name:<12} configured={info.get('configured')} models={','.join(info.get('models', [])[:1])}")
        latest = c.get(f"{TELEMETRY}/latest").json()
        window = c.get(f"{TELEMETRY}/window?seconds=8").json()
        telemetry = {
            "signals": latest.get("signals", {}),
            "mean": window.get("mean", {}),
            "trending": window.get("trending", {}),
            "mode": latest.get("driving_mode", "normal"),
            "charging": latest.get("charging", False),
            "warnings": [w.get("message") for w in latest.get("warnings", [])],
            "dtcs": [d.get("code") for d in latest.get("active_dtcs", [])],
        }
        for q in ("What is my remaining range?", "Do I have any active diagnostic trouble codes?"):
            docs = c.post(f"{RAG}/retrieve", json={"query": q, "top_k": 3}).json().get("top_k", [])
            r = c.post(
                f"{GATEWAY}/v1/complete",
                json={
                    "messages": [{"role": "user", "content": q}],
                    "grounding": {"telemetry": telemetry, "documents": docs, "events": []},
                },
            )
            data = r.json()
            print(f"  asked: {q!r}")
            print(f"    -> [{data.get('provider')} / {data.get('model')}] {data.get('content', '')[:180]}...")
            print(f"       latency={data.get('latency_ms')}ms cost=${data.get('cost_usd', 0):.5f} cache={data.get('cache_hit')}")
        stats = c.get(f"{GATEWAY}/stats").json()
        print("  totals:", {k: stats.get(k) for k in ("calls_by_provider", "spend_usd_by_provider")})


def demo_assistant() -> None:
    _banner("4 · ASSISTANT API  (grounded answers · citations · refusal)")
    with _client() as c:
        rows = [
            ("How is my battery doing?", "battery-health"),
            ("Do I have any active diagnostic trouble codes right now?", "diagnostics"),
            ("Give me a status summary.", "summary"),
            ("Tell me a joke.", "refusal"),
        ]
        for q, tag in rows:
            r = c.post(f"{ASSISTANT}/v1/ask", json={"question": q})
            data = r.json()
            state = "REFUSED " if data.get("refused") else "answer "
            print(f"  [{tag:<12}] {state}-> {data['answer'][:200]}\n       cites={len(data['citations'])} intents={data['intents']} "
                  f"provider={data['provider']}")


def demo_monitoring() -> None:
    _banner("5 · MONITORING  (PSI/KS drift detection vs baseline)")
    with _client() as c:
        d = c.get(f"{MONITOR}/monitor/drift").json()
        print(f"  drift status: {d['status']}  alerted={d['alerted']}  baseline={d['baseline_source']} ({d['baseline_frames']} pts)")
        print("  per-signal PSI:", {k: v for k, v in d.get("alert_signals", {}).items()})

    _banner("6 · INJECT A DRIFT SCENARIO  (biased speed/SOC distribution)")
    with _client() as c:
        c.post(f"{TELEMETRY}/api/scenario", json={"type": "drift", "duration_s": 30, "intensity": 0.7})
        print("  scenario=drift activated for 30s — watching the detector for ~20s")
        for i in range(8):
            time.sleep(2.5)
            d = c.get(f"{MONITOR}/monitor/drift").json()
            print(f"    t+{2.5 * (i + 1):>4.0f}s  status={d['status']:<14} overall_psi={d.get('overall_psi', 0):7.2f}")
            if d.get("alerted"):
                print("    -> DRIFT ALERT fired — model retraining / rollback trigger valid")
        sh = c.get(f"{MONITOR}/service-health").json()
        print("  fleet health:", sh)


def demo_safety() -> None:
    _banner("4b · SAFETY POLICY  (read-only / restricted / prohibited)")
    with _client() as c:
        rows = [
            ("What is my battery level?", "read_only (allowed to LLM)"),
            ("Start charging the vehicle now.", "restricted_action (needs confirmation)"),
            ("Disable the battery thermal protection right now.", "prohibited_action (deterministic refusal)"),
            ("Help me reflash the ECU to remove the speed limiter.", "prohibited_action (deterministic refusal)"),
        ]
        for q, expect in rows:
            r = c.post(f"{ASSISTANT}/v1/ask", json={"question": q}).json()
            print(f"  [{r.get('policy_level', 'read_only'):<20}] {q!r}")
            print(f"       refused={r.get('refused')} provider={r.get('provider')} role={expect}")


def demo_gate_reminder() -> None:
    _banner("7 · EVAL GATE (what CI actually blocks on)")
    print("  golden set:  16 cases · replay-grounding · aurora-prompt-v9 · rag-index-v4")
    print("  gate:        pass_rate >= 0.80 (signal_accuracy, doc_grounding, refusal_honesty, policy_compliance)")
    print("  run it:      make eval-e2e     # local")
    print("               docker compose run --rm eval --assistant http://assistant-eval:8100   # container")


def main() -> int:
    _banner("AURORA · Production In-Vehicle AI Assistant Platform — DEMO")
    from vehicle_ai.version import PROMPT_VERSION, RAG_INDEX_VERSION, TELEMETRY_SCHEMA_VERSION, __version__

    print(f"  platform v{__version__} · prompt {PROMPT_VERSION} · rag {RAG_INDEX_VERSION} · telemetry-schema {TELEMETRY_SCHEMA_VERSION}")
    check_health(8100, "assistant-api")
    check_health(8101, "telemetry-simulator")
    check_health(8102, "llm-gateway")
    check_health(8103, "rag-service")
    check_health(8104, "monitoring")

    if sys.argv[1:] == ["--quick"]:
        return 0

    demo_telemetry()
    demo_rag()
    demo_gateway()
    demo_assistant()
    demo_safety()
    demo_monitoring()
    demo_gate_reminder()
    _banner("DEMO COMPLETE — stack is live and green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())