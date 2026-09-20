# WS5 — Cockpit + browser voice experience (design, for review)

Status: **IMPLEMENTED** (WS5 approved with decisions D1–D4; see §8 for what
shipped and how to review it). The plan below is the approved design.

Every surface we plan consumes *existing, already-frozen contracts* — no new inference anywhere. The cockpit and the browser app are **one web app in two
presentation modes** (glance distance vs desk). Nothing here is a generic admin
dashboard: it is an EV-native driver + owner experience.

---

## 8. Implementation delivered

**D1 — `GET /v1/hud` facade.** `services/assistant-api/app/api/hud.py` +
route in `app/api/routes.py`. Reads-only aggregation of frozen state from its
owners (assistant telemetry buffer, WS4 engine, prediction service status,
gateway `/providers`+`/routes`, rag `/index-info`, per-service `/healthz`).
No inference: every number carries a `provenance` from the five-way taxonomy
(`measured` / `prediction` / `derived` / `documentation` / `simulator`), absent
signals are `present:false` with `value:null`, unreachable owners surface an
explicit `error` label. `?prediction=1` lazily fetches the deterministic
explanation. Unit-covered in `tests/test_hud.py` (7 tests; suite 211 passed).

**D2 — Voice.** Push-to-talk via `SpeechRecognition` + typed fallback + TTS
(`speechSynthesis`), op-in "Voice on". Voice speaks only grounded answers and
code-decided alerts (`/v1/proactive/ask`); refusals are spoken calmly. Wake
word / Vosk / VAD are deferred as decided.

**D3 — One app.** `services/assistant-api/app/static/` → `index.html` +
`app.css` + `app.js` + `fixture.json`. `?mode=cockpit` enables the glance-first
driving layout (larger numerals, Drive/Alerts/Ask only, Inspect hidden);
`?fixture` renders the whole experience from `fixture.json` with the stack
down — the design is reviewable offline.

**D4 — Base vs LoRA in the Inspect panel** (`view-inspect -> Compare`), not the
driver surface: one prompt to `peft-base` and `peft` via gateway
`/v1/debug/prompt` + `/v1/complete`, side-by-side, refusal states highlighted.

**Live e2e driving scenario** (sim :8101, assistant :8100): trip started
(destination 92 km) → prediction `status: active` (coverage 21%, λ=1.0, ranked
causes, identity checks pass) → mid-drive leak (`/api/scenario` leak 1.0) →
4× `tire_pressure_leak@warning` with slope triggers (`*_slope_per_min` in
`kPa/min`) and **separate** evidence rows (pressure kPa / rate kPa/min),
`prediction_discharge_risk` honestly signal-gated at 2205 frames → worded via
`/v1/proactive/ask` (`events_used` matches the decided event) → ack removes the
event from `active`. Artifacts: `artifacts/ws5/hud-scenario.json`,
`proactive-snapshot.json`, `ask-tyres.json`.

## 1. Design language (EV-native, not admin)

- **Single-feature-per-card, big-serif numerals.** SOC is a donut, range is a
  ring around it, charge is an arc, tyre pressure is a 4-quadrant gauge, pack
  temperature is a thermal strip. Actual numbers are primary; the eye reads
  status in one glance and detail on demand.
- **Dark, near-black theme with ONE accent channel** (existing `--acc` teal)
  and a strict severity palette reserved for meaning (`ok`/`warn`/`crit` +
  `info`), not decoration. Ambers/reds never appear just to look busy.
- **No unexplained numbers.** Every numeral on screen has a provenance path:
  click it (or tap it, or ask about it) and a slide-over shows *how the number
  was derived* — telemetry → feature → physics/ML split → model/adapter
  version → retrieval tier/edition/page when text is involved.
- **Honesty as a visual state, not a failure.** `status=physics_only_fallback`,
  `provider=mock/base`, `policy_level=prohibited`, and refusals all render with
  the same calm, explicit label they carry in the data contract. Nothing is
  decorated to look like success or made to look like a crash.
- **Reading distance.** Cockpit mode uses ≥110%-scale type, 3 s glance rhythm,
  no hover-dependent state; browser mode relaxes both.

## 2. Data contract (all existing endpoints, composed client-side)

The web app aggregates server-side contracts as-is — no new aggregation service
in v1 (open decision D1). Sources:

| surface | source | payload |
|---|---|---|
| Live cards | assistant `GET /v1/wire/live` | compact signals, charging, DTC codes, warnings, source/mode |
| Signal series (sparklines) | sim `GET /window` + `GET /latest` | windowed mean/min/max/std per signal |
| Proactive alerts | assistant `GET /v1/proactive` + `POST /v1/proactive/ask` | snapshot: active/recent, eligible vs skipped, poll interval; ack via ask `ack_event_ids` |
| Prediction + explanation | sim `POST /api/prediction/explain-from-telemetry`, `GET /api/prediction/status` | `PredictionExplanation` (physics split, ML residual, λ, ranked causes, identity checks, status) |
| Trip / what-if | sim `POST /api/trip/arrival-estimate`, `energy-drivers`, `what-if` | arrival SOC, driver split, directional deltas |
| RAG provenance | rag `POST /retrieve` (+ assistant citations/grounding on `/v1/ask`) | top_k with tier, edition, page, source_id, vehicle_match, rights_status |
| Base/LoRA | gateway `GET /providers`, `/stats`, `POST /v1/debug/prompt` | provider/adapter identity, healthy chain, base-vs-lora same-prompt responses |
| Safety state | assistant `/v1/ask` response (`policy_level`, `refused`, `safety_blocked`) | per-answer policy level + refusal reasons; proactive `CRITICAL_DTC_SYSTEMS` |
| Platform health | monitoring `GET /service-health`, `/ai-quality`, `/monitor/drift` | service ready states, live counters, drift PSI |
| Identity footer | assistant `GET /v1/environment` | service/version, prompt v9, rag v4, environment |

## 3. Screens

**Home · Live (glance)** — SOC donut inside a range ring with a charge arc
when plugged; energy card (motor/regen/aux power), thermal strip (pack/coolant
with the 46/54 °C thresholds as ticks), 4-quadrant tyre gauge with per-wheel
pressure + slope sparkline when a leak detector is active; outside temp. Top
banner = top proactive alert (code-decided; tap → alert card). Safety pill =
current `policy_level`.

**Alerts** — proactive active/recent timeline. Each card shows severity, `status`
(full / physics-fallback), **conditions verbatim** (e.g. trigger
`tire_pressure_rr_kpa_slope_per_min observed=-22.61 kPa/min vs threshold -12.0`)
and the **separate evidence fields** (pressure `220.1 kPa`, rate `-22.61
kPa/min`), first/last seen, `occurrence_count`, ack. A "detectors eligible vs
skipped" strip shows the current data-availability gate — missing signals render
as *why no alert is (yet) shown*, not as silence.

**Predict · Explain** — arrival-estimate card plus energy-driver split (aero /
rolling / inertia / aux) and what-if sliders (Δspeed, Δcabin temp) recomputing
directional deltas. The explanation slide-over renders the *whole* deterministic
`PredictionExplanation`: physics vs ML-residual bar, ranked causes with
`provenance=physics|ml`, λ applied, identity checks, model/adapter version,
`status` badge. Physics-fallback states label the estimate as such and never
show ML attribution.

**Ask · Provenance** — the existing chat panel, upgraded: every answer card
shows `provider/model/latency/intents` (already there) plus `policy_level` and,
when documents grounded the answer, citation cards with tier, edition, page and
`rights_status` — knowledge-pack citations tagged **"internal assistant guide —
not manufacturer documentation"**, manual/appendix tagged with pages. Tapping a
citation opens the retrieval evidence. Refusals render the `safety_blocked`
reason verbatim with a calm tone and a safe alternative (e.g. "I can't disable
tyre-pressure monitoring — here's the current pressure instead").

**Models · Base/LoRA** — gateway provider table (`/providers`): which provider
serves the assistant, adapter identity (r/α, method `lora-r8-alpha16`,
distilled vs real), `healthy_chain`. A **side-by-side compare** posts one
prompt to base and LoRA (`/v1/debug/prompt`, `/v1/complete` on each `peft`
provider) and shows both outputs + gateway refusal behavior. This is the honest
"was the adapter worth it" screen, not a benchmark dashboard.

**Safety** — read-only/review screen: current `policy_level`, recent refusal
log with reasons (`safety_blocked`), safety-critical DTC list (restraints /
braking / high voltage), and the proactive CRITICAL alerts paired with the
policy state. Crew/engineers can probe "what would AURORA refuse" and see base
vs LoRA refusal differences — the same measurement the gateway router already
describes at `routes.py` refusal handling.

## 4. Voice experience

Two stacked modes in the same web app:

- **Browser (desktop demo):** push-to-talk + optional on-device wake word
  ("Hey AURORA"). ASR → the existing `POST /v1/ask` pipeline (safety gate,
  grounding, RAG, policy) → TTS of the grounded answer text. Zero new backend.
- **Cockpit:** continuous mic, VAD gating, **barge-in** (mic listens while an
  answer speaks; a new utterance cancels the current TTS and re-queries), noise
  gate tuned for cabin. Same `/v1/ask` path.

**Voice rules (hard, reusing existing contracts):**
- Voice only ever speaks: (1) grounded answers from `/v1/ask`, (2) code-decided
  proactive alerts from `top_active`. It never generates its own claims.
- Proactive voicing respects the same honesty as the UI: `status=fallback`
  alerts are spoken with "this is a physics estimate, the model isn't
  available"; ack on "OK".
- Refusals are spoken as short, calm refusals + a grounded alternative, never
  as technical failures.
- TTS reads the *response text*, not the prompt. Citations can be read aloud
  only when asked; by default voice says the answer, the browser highlights the
  citation card simultaneously (multi-modal, no lip-sync).

**ASR/TTS choices (open decision D2):** demo-default = Web Speech API (synthesis
+ recognition, zero dependency); offline variant = Vosk/WASM ASR + local TTS for
the cockpit promise. Wake word on-device (OpenWakeWord) — licensing flagged.

## 5. Implementation shape (for the next phase, not started)

1. **Theme + fixture mode:** rebuild `app/static/index.html` with the language
   above and a `?fixture` mode that renders the whole experience from checked-in
   deterministic JSON — the design and interactions are reviewable with the
   stack down.
2. **Live wiring:** swap fixtures for the real endpoints (§2); `/v1/proactive`
   2 s poll + alert banners; ask panel citations + provenance slide-over.
3. **Prediction + Models screens:** explanation slide-over; base-vs-LoRA compare.
4. **Browser voice:** PTT → `/v1/ask` → TTS; citation highlight during speech.
5. **Cockpit mode:** glance layout + barge-in VAD; reviewed on a big display.

Single-file web app, no new build tooling or infra; served by the existing
assistant static mount.

## 6. Acceptance gates for the plan (when implemented)

- A grounded answer's citation card shows tier/edition/page + rights_status;
  knowledge-pack is labelled non-manufacturer.
- A prediction's explanation slide-over shows physics vs ML, ranked causes,
  λ, identity checks, and the fallback badge on `physics_only_fallback`.
- A proactive alert card shows verbatim conditions + separated pressure/rate
  evidence; ack works and the event leaves `active`.
- Base vs LoRA compare answers the same prompt and shows which adapter served
  the assistant.
- A prohibited request renders `policy_level` + reason verbatim and never as a
  crash; safety-critical DTC list is visible.
- Voice: "what's my tyre pressure doing?" answers from live telemetry; a
  code-decided alert is voiced, fallback wording respected; barge-in kills the
  current spoken answer.

## Open decisions for review

- **D1:** client-side aggregation (recommended, keeps WS5 backend-free) vs a
  `GET /v1/hud` facade endpoint on the assistant for fewer browser fetches.
- **D2:** ASR/TTS — Web Speech (zero-dep demo) vs Vosk/WASM offline for the
  cockpit promise; wake-word engine + license.
- **D3:** cockpit = the same web app with a `?mode=cockpit` (recommended) vs a
  separate thin client; both are reviewable in this repo.
- **D4:** where Base/LoRA side-by-side lives — gateway GUI vs a page in this
  app (recommended: this app, so the comparison shows *in context*).

## Scope boundaries

- No Android/Automotive integration; the cockpit is a web rendering target.
- No new backend services, no model work, no new inference contracts.
- TTS quality is POC/demo-grade in this phase; production TTS is out of scope.