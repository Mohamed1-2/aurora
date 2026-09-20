# AURORA — Architecture

## 1. Guiding principles

1. **Ground everything.** No answer without provenance. The assistant only speaks from
   (a) the vehicle's live aggregated telemetry and (b) retrieved manual passages, each
   referenced by `【doc：section】` citation markers. Unanswerable ⇒ honest refusal.
2. **Deterministic where it matters.** Simulator seed, RAG chunking/scoring, golden
   dataset, eval metrics and drift baseline are all reproducible — that is what makes
   CI gating trustworthy.
3. **Offline-first, cloud-capable.** The provider chain degrades gracefully and the
   platform is fully functional with the grounded mock provider.
4. **Contracts in code.** All wire formats live in `shared/vehicle_ai/schemas.py`;
   services evolve independently while pydantic enforces compatibility.

## 2. Telemetry path

The **telemetry-simulator** emits `TelemetryFrame`s every 100 ms (10 Hz) over WebSocket
and exposes REST views:

* `/latest`, `/window?seconds=15` (mean/min/max/std), `/signals` (catalog + units)
* `/api/scenario` — injects **leak / dtc / drift / charge / battery_heat** conditions
* `/metrics` (Prometheus) — frames produced, subscribers, scenario counters

The engine is a physics-lite EV model: aero + rolling + inertia loads, 82 kWh battery,
regen recovery, HVAC draw, thermal mass on coolant/battery, Wh/km gating (60–520, only
above 9 km/h) and per-scenario parameterisation. Signals are scaled/quantized to look like
a real vehicle bus and include **pack voltage & current, cabin temp + setpoint, odometer,
trip distance + energy, and coherent GPS** — the full context a vehicle assistant reasons over.

## 3. Ingestion & aggregation (assistant-api)

`services/assistant-api/app/telemetry/ingest.py`:

* live mode: connects to `ws://telemetry-simulator:8101/ws`, buffers frames
* replay mode: `AURORA_ASSISTANT_TELEMETRY_REPLAY=<snapshot.json>` — deterministic
  ingestion used by evaluation and offline CI
* `aggregate_window()` derives mean/min/max/std + trending over rolling windows
* a loss of stream (with heartbeat timeout) auto-falls back to replay if configured

## 4. Grounding construction

For a question `q` the assistant:

1. classifies **intents** (`app/core/intents.py`) — range, battery, charging, tire,
   diagnostics, temperature, energy, winter, maintenance, speed, driving_mode, status, hello
2. runs the **safety policy gate** (`app/core/policy.py`) — every request is tiered
   `read_only / safe_action / restricted_action / prohibited_action` **before any LLM call**.
   Prohibited and restricted requests get a deterministic answer and never touch the LLM.
3. retrieves KB chunks → `rag-service/retrieve` with per-intent **doc weights**
   (e.g. diagnostics ⇒ “05-diagnostics-and-dtc.md” boosted)
4. builds a **prompt context** (`app/core/prompt.py`) marked with `\n\n## ` sentinel:
   live signal snapshot, DTCs/warnings, retrieved docs, system prompt + question.
   Providers split on the sentinel so intent detection never runs on the context.
5. calls the **llm-gateway** `/v1/complete`, parses citation markers back into
   `citations[]`, derives a risk level from temperatures/pressures/DTCs, records AI-quality
   counters and returns an `AssistantResponse` (with the resolved `policy_level`).

### Safety policy layer

```
 user question
      │
      ▼
 intent classification
      │
      ▼
 safety policy ──► read_only ───────► grounded LLM
      │              safe_action ────► grounded LLM
      │              restricted ─────► confirmation-required (deterministic)
      └────────────── prohibited ────► deterministic refusal      (never the LLM)
```

Patterns are compiled regexes, first-match-wins. Prohibited coverage: safety-system
tampering (brake/ABS/airbag/TPMS/traction), battery management bypass, driver-assistance
disable, autonomous-driving engagement, speed-limit override, firmware/ECU reflash,
data exfiltration, and electronics compromise. Restricted: charging start/stop, climate
control, door/trunk/lock, preconditioning. This is the architectural boundary that makes
it a *vehicle* assistant rather than a chatbot, and the golden set hard-checks it via
`policy_compliance`.

## 5. LLM gateway

`app/router.py` runs the provider chain:

* ordered **priority** `azure-openai, openai, mock` with per-provider **circuit breaker**
  (3 failures → open 30 s → half-open probe)
* **TTL cache** on `cache_key` (defaults to content hash) — eval/CI runs hit the cache
* **spend ledger**: cost per 1k tokens × usage per provider, exposed at `/stats`
* **safety gate** on every completion (blocked topics incl. “joke”) → `refused`

The **mock provider** (`providers/mock.py`) is the workhorse: intent dispatch table,
template handlers that read the *actual* signals, honest refusal, and citation markers.
Fully offline and deterministic — the base of CI, eval and offline demos.

## 6. RAG

`app/index.py` — pure-function BM25 (`k1=1.5, b=0.75`) over the provenanced manual
corpus (tier/audience-aware) layered with the knowledge pack. The index content hash is
exposed (`/index-info`) and becomes `rag-index-v4`; an
evaluation run records which index/prompt it scored against. Retrieval is boosted by
search-position-doc weights from the assistant.

## 7. Evaluation gate

`services/evaluation/`:

* **golden set** — 16 cases (`datasets/golden.yml`): 12 in-scope + 1 out-of-scope refusal
  + 3 adversarial **safety-policy** cases (prohibited / restricted / prohibited-firmware)
* **strategy** — replay the deterministic `telemetry_snapshot.json`, ask each case, grade:

  | metric | what it tests | weight |
  |---|---|---|
  | `signal_accuracy` | quoted live numbers within tolerance | 0.45 (hard ≥ 0.5) |
  | `signal_presence` | expected signals actually used | – |
  | `doc_grounding`   | citations point at expected docs | 0.20 (hard ≥ 0.3) |
  | `text_recall`     | lexical overlap with reference | 0.25 |
  | `intent_match`    | intent classification agreement | 0.10 |
  | `refusal_honesty` | out-of-scope ⇒ refusal | – |
  | `policy_compliance` | safety tier + refusal correct | – |

  `grade = Σ weights; passed = grade ≥ 0.55 and all hard thresholds`.
  **Gate: pass_rate ≥ 0.80** (current standing: **1.00**).
* **MLflow** client (`framework/mlflow_client.py`) logs the report, artifacts and model
  when `MLFLOW_TRACKING_URI` is reachable — gracefully skipped otherwise.
* entrypoint `run.py` also **registers** the accepted model snapshot.

### 7a. Fine-tuning a domain LoRA

A build-time step (`services/evaluation/finetune.py`, Colab `colab/train_lora.ipynb`)
trains a **rank-8 LoRA** on `q_proj`/`v_proj` over curated, grounded `train/val/adversarial`
JSONL. `compare.py` then runs the **same** golden set against the baseline vs the
adapter-serving assistant and requires **beat-or-tie on every grounded metric including
`policy_compliance`** — CI blocks on regression. See `docs/finetuning.md`.

## 8. Monitoring / drift

`app/drift.py` — two-stage test vs `baseline.json` (6000 pts/signal, seed 42):

* **PSI** (population stability index, 10 bins) — distribution shift
* **KS** — Kolmogorov–Smirnov p-value — shape shift with small samples

`/monitor/drift` returns per-signal status; alerts fire when `psi ≥ 0.25` or
`ks_p ≤ 0.01`. `/service-health` polls the fleet (`/readyz`), `/insights` aggregates.
Prometheus counters feed Grafana if configured.

## 9. Reproducibility table

| artifact | source of truth | repinned by |
|---|---|---|
| telemetry window | live `ws://…/ws` or snapshot | `AURORA_ASSISTANT_TELEMETRY_REPLAY` |
| RAG behaviour | corpus chunks + BM25 pure function | `rag-index-v4` index hash |
| prompt | prompt builder versioned | `aurora-prompt-v9` |
| proactive alerts | signal-gated detectors + state machine | `proactive-policy-v1` |
| drift baseline | `make baseline` (seed 42) | `baseline.json` |
| eval golden set | `datasets/golden.yml` | committed with every run |
| eval run | `run.py` (run-id, dates, versions) | report JSON + MLflow |

## 10. Deployment topology

* **Compose** (dev/CI): one container per service, healthchecks, `profiles: [eval]`
  for the replay-grounded gate job; all images built from `shared/` editable install.
* **Kubernetes/AKS** (`infra/k8s`): deployments + ClusterIP services per component,
  HPA on assistant-api, ingress (nginx, TLS), configmap for non-secret env, secret
  template for provider keys, nightly eval-gate CronJob.
* **Terraform** (`infra/terraform`): RG → ACR → KeyVault → storage (MLflow) → AKS.
* **CI** (`.github/workflows`): `lint → test → build&push → eval gate → deploy`.