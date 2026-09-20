# WS4 — Proactive intelligence: alerts before the driver asks (implemented)

Status: **FROZEN.** RAG frozen; WS4 hard finding shipped (see §10) and the
golden evaluation re-passed. Records the deterministic detection layer, the
surfacing channels, results of the golden scenario evaluation, and one
end-to-end live example.

---

## 1. Principle: "code decides existence, the LLM only words it"

AURORA's proactive layer has two strictly separated stages:

1. **Detection (code, deterministic)** — `ProactiveEvaluator` runs a state
   machine over the assistant's live 10 Hz telemetry buffer. It decides
   *whether* an alert exists, its severity, and why (via `ProactiveTrigger`s:
   signal, observed value, threshold, direction, window, samples). No LLM is
   involved; the behavior is diff-able testable policy in
   `app/core/proactive_policy.py`.
2. **Wording (LLM, non-authoritative)** — `POST /v1/proactive/ask` hands the
   already-decided `ProactiveEvent`s to `build_proactive_messages`. The prompt
   rules are hard restrictions: quote trigger evidence verbatim, never invent
   a new alert, `status=fallback` must say "physics estimate", never attribute
   physics to ML, never present the knowledge-pack as manufacturer docs. If the
   engine reports no events the endpoint returns a deterministic
   "No active vehicle alerts right now." **without any LLM call.**

Surfacing = two approved channels (a)+(b):

- **(a) `GET /v1/proactive`** → `ProactiveSnapshot`. Code-only JSON: active
  events, recent history, which detectors were eligible vs signal-gated this
  poll, poll interval. For dashboards/UI without an LLM.
- **(b) `POST /v1/proactive/ask`** → `AssistantResponse` wording of the top
  `count` alerts. `ack_event_ids` dismisses alerts until the event resolves
  and recurs.

WebSocket push stays deferred to the cockpit workstream.

---

## 2. Contract — strict signal gating

Every detector declares the signals it requires. **If any required signal is
absent at a poll, the detector is skipped, produces no inference and absolutely
no alert**; the skip is reported in
`detectors_skipped_no_signal` (per-detector missing-signal lists). This applies
to TPMS (all four wheel pressures for the leak detector), range, DTC field
presence, charging-power, and the six
`PREDICTION_INPUT_SIGNALS` (`vehicle_speed_kph`, `battery_soc_pct`,
`outside_temp_c`, `coolant_temp_c`, `battery_voltage_v`, `hvac_power_kw`) that
build the 300 s SOC-drop feature vector — the discharge-risk detector can never
fire when the window cannot be built.

## 3. Detector catalog (policy v1)

| detector | gate (required signals) | condition → severity | confirm | cooldown | resolve |
|---|---|---|---|---|---|
| `low_range` | `distance_to_empty_km` | < 40 km (suppressed while charging) → warning | 2 polls | 600 s | 3 polls |
| `tire_pressure_leak` | all four `tire_pressure_*_kpa` | sustained slope < −12 kPa/min AND drop > 3 kPa over 30 s, per wheel → warning | 2 | 900 s | 3 |
| `battery_overtemperature` | `battery_temp_c` | 15 s mean > 46 °C → warning; > 54 °C escalates to critical | 2 | 900 s | 3 |
| `charging_stall` | `charge_rate_kw` | charging true AND 20 s mean < 0.1 kW → warning | 2 | 1200 s | 3 |
| `dtc_escalation` | `active_dtcs` field present | new active code → warning/critical by catalog; severity jump re-emits immediately | 1 | 600 s | 3 |
| `prediction_discharge_risk` | `PREDICTION_INPUT_SIGNALS` | ≥ 3000 frames (full 300 s) AND calibrated SOC drop ≥ 4.0 → info; `status=fallback` if physics estimate | 2 | 600 s | 3 |

Guardrail from the approved plan: the calibrated 300 s SOC-drop is surfaced
**only** as a discharge-risk / high-consumption alert — it is **never** turned
into an arrival-SOC claim. `prediction_discharge_risk` is the single detector
that consumes the promoted prediction system; it reuses the explain-from-
telemetry fetch path already used for WS3.

## 4. Lifecycle / state machine

One `_EventState` per `(detector, key)` (keys: `range`, wheel, `battery`,
`charge`, DTC code, `discharge`), stable `event_id = sha1(detector:key:vehicle)[:12]`.

- **Probation** — candidates must persist `confirm_polls` consecutive polls
  before the first emission (noise guard).
- **Active / duplicate suppression** — while active, re-emissions are blocked
  inside `cooldown_s` (stats `suppressed`); once cooldown elapses with evidence
  persisting, a repeat emission is made (`occurrence_count += 1`).
- **Monotonic severity** — a severity escalation bypasses cooldown and
  re-emits immediately (e.g. battery 46→54 °C).
- **Resolution** — after `resolve_after_polls` polls with no evidence, the
  event moves to `resolved`; it leaves `active`.
- **Recurrence** — a new episode after recovery re-arms the same state and
  re-warns the driver (`occurrence_count += 1`). Occurrence count is therefore
  an episode+repeat counter, not a wall-clock log.
- **Ack** — `ack(event_ids)` marks events dismissed; dismissed events leave
  `active` until the underlying event resolves and recurs.

The evaluator accepts an injectable `clock`, which the golden evaluation uses
to make cooldown/repeat behavior deterministic.

## 5. Wiring

- `services/assistant-api/app/main.py` — lifespan builds the `ProactiveEvaluator`
  over the shared `TelemetryIngest` (prediction fetch = the pipeline's WS3
  explain-from-telemetry call) and runs a background `_proactive_loop` at
  `proactive_poll_interval_s` (5 s). Eval vehicle id `ORION-X-TEST-01`.
- `app/api/routes.py` — `GET /v1/proactive`, `POST /v1/proactive/ask`
  (ack applied before worded); both 503 if the evaluator is disabled.
- `app/core/service.py::word_proactive` — deterministic empty-event
  short-circuit, else `build_proactive_messages` → gateway complete call,
  `risk_level` from top event severity, `outcome="proactive"`.
- `shared/vehicle_ai/schemas.py` — `ProactiveTrigger`, `ProactiveEvent`,
  `ProactiveSnapshot`, `ProactiveAskRequest`.
- Config: `proactive_enabled`, `proactive_poll_interval_s`, `proactive_max_alerts`.
  Prompt version bumped to `aurora-prompt-v9`.

## 6. Golden scenario evaluation (`scripts/eval_proactive.py`)

Deterministic in-process replay of scripted 10 Hz timelines with an injected
clock; expectations in `services/evaluation/proactive/golden.yml`.

```
scenario            emissions                                                    active
none                —                                                            []
leak                tire_pressure_leak@warning(t=20,occ=1,full)                   ['tire_pressure_leak']
battery_heat        battery_overtemperature@warning(t=295), @critical(t=450)      ['battery_overtemperature']
range               low_range@warning(t=5,occ=1), @warning(t=605,occ=2)           ['low_range']
dtc                 dtc_escalation@critical(t=15,occ=1,full)                      ['dtc_escalation']
charge_healthy      —                                                            []
charge_stalled      charging_stall@warning(t=5,occ=1,full)                        ['charging_stall']
discharge           prediction_discharge_risk@info(t=305,occ=1,full)              ['prediction_discharge_risk']
discharge_fallback  prediction_discharge_risk@info(t=305,occ=1,fallback)          ['prediction_discharge_risk']
```

**Metrics: precision 1.0, recall 1.0, 8/8 expected events, 0 false alerts,
0 false-alert runs.** Lifecycle checks all pass: leak stays a single emission
inside the 900 s cooldown; low-range suppresses duplicates inside 600 s then
re-emits at 605 s (occ 2); battery escalates warning→critical monotonically;
DTC never repeats inside its cooldown; discharge fires only after ≥ 300 s
coverage is present (first at t=305) and the fallback run stays `status=fallback`
at info severity. Report: `artifacts/proactive/eval/report.json`.

## 7. End-to-end live example (eval stack, mock gateway)

Stack: telemetry-simulator `:8101` (live WS), llm-gateway `:8102` (mock),
rag `:8103`, assistant `:8100` (live feed).

Trigger `POST /api/scenario {"type":"leak","duration_s":240,"intensity":1.0}`
(rr wheel at 1.75 kPa/s). ~40 s later (30 s window + 2 confirm polls):

```
GET /v1/proactive
ACTIVE (1)
  tire_pressure_leak@warning occ=1
    Pressure on rr is falling at 21 kPa/min (10.9 kPa over 30 s).
    trigger tire_pressure_rr_kpa observed=-20.68 dir=below threshold=-12.0 kPa/min
```

Cooldown suppression observed live: 60 s later the same event is still active,
single `occurrence_count`, no duplicate emission. Then `POST /v1/proactive/ask`
(call the wording layer): response `AssistantResponse` with
`grounding.events_used=[rr event id]`, `citations=["Tyre pressure drop"]`,
`risk_level=warning`, `model=aurora-titan-e-local` (mock). The mock gateway
returns its canned greeting instead of LSTM-composed prose, so the demo proves
the full **detection → top-active → grounding → risk-level → response plumbing**,
not final prose quality.

`ack_event_ids=[<event id>]` dismisses the alert: it leaves `active` and the
next ask returns `risk_level=info`.

## 8. Known live-side observations (eval simulator)

- The simulator's tire model mean-reverts all four wheels and couples noise, so
  `fl/fr/rl` can show leak-like slopes identical to `rr` during a rr leak. The
  detector correctly reports each wheel whose own signal crosses threshold —
  a sim artifact, not a detector bug. On real per-wheel telemetry only the
  leaking wheel fires.
- The simulator streams `battery_temp_c` around 44–53 °C, i.e. above the 46 °C
  Policy-v1 ceiling, so `battery_overtemperature` can be persistently active on
  the eval vehicle. Thresholds are reviewable constants; the live observation
  is a placeholder-data artifact, and the code chooses the detector's hard facts
  (mean, limit, window) from `proactive_policy.py`.
- The sim `signal_catalog` lacks `hvac_power_kw`, so `prediction_discharge_risk`
  is correctly signal-gated out on the live feed (visible in
  `detectors_skipped_no_signal`). The golden eval covers the discharge detector
  deterministically instead.

## 10. Hard finding shipped before freeze — trigger evidence semantics

Finding: the tyre-leak trigger exposed `signal="tire_pressure_rr_kpa"` while its
`observed` was a **rate** (−22.61) with unit `kPa/min` — the signal name implied
absolute pressure (kPa) but the value was a slope. Dimensional mismatch in the
alert contract, and a trap for any UI/wording layer that trusted the signal name.

Fix (semantics-first, per review):
- The trigger now names the **rate variable**: `tire_pressure_rr_kpa_slope_per_min`,
  `observed=-22.61`, `threshold=-12.0`, `unit="kPa/min"`, `direction="below"`
  — a slope represented as a slope.
- Raw pressure and change rate are carried as **separate `evidence` fields**
  on the event (`ProactiveEvidence`):
  - `tire_pressure_rr_kpa = 220.1 kPa (current pressure)`
  - `tire_pressure_rr_kpa_slope_per_min = -22.61 kPa/min (pressure change rate)`
- Wording layer renders conditions and observed quantities as distinct rows and
  quotes them verbatim.
- Regression tests: `test_tire_leak_trigger_is_a_slope_not_a_pressure` asserts the
  rate signal name, the `kPa/min` unit, threshold `-12.0`, and that pressure +
  rate live in separate evidence entries.

Verified after the fix: **204 unit tests pass, golden gate PASSED** (precision
1.0, recall 1.0, 0 false alerts), and the live eval stack emits exactly the
shape above for a triggered rr leak.

## 9. Files

- `services/assistant-api/app/core/proactive_policy.py` — detector catalog (policy v1).
- `services/assistant-api/app/core/proactive_engine.py` — pure detectors + state machine.
- `services/assistant-api/app/core/prompt.py` — `build_proactive_messages` wording layer.
- `services/assistant-api/app/core/service.py`, `app/api/routes.py`, `app/main.py` — wiring.
- `tests/test_proactive_engine.py` — 21 unit tests (gating, thresholds, slope-vs-
  pressure trigger semantics, fallback, cooldown, escalation, resolution,
  recurrence, ack).
- `scripts/eval_proactive.py`, `services/evaluation/proactive/golden.yml` — golden scenario gate.
- `artifacts/proactive/eval/report.json` — accepted evaluation.