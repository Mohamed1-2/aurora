# AURORA · Production In-Vehicle AI Assistant (MLOps Platform)

**Brand** AURORA · **Vehicle** AURORA · an end-to-end, physically-grounded AI assistant
platform for electric vehicles — from simulated telemetry stream to LLM gateway with
provider failover, RAG grounding, offline evaluation gating, drift monitoring, containerized
deployment (Docker / Kubernetes / Azure) and a CI/CD pipeline.

> Runs **with zero cloud**. A grounded, deterministic, always-available mock provider answers
> from the real vehicle telemetry and knowledge base; OpenAI / Azure OpenAI are first-class
> chain members whenever keys are provided.

---

## What it does

Ask your car a question, in a plain sentence:

```text
> What is my current range and state of charge?
Estimated remaining range is **201 km** at 62% charge. ...
> Do I have any active diagnostic trouble codes?
No active diagnostic trouble codes in the telemetry stream right now. ...
> Tell me a joke.
I don't have reliable data to answer that from the vehicle stream or the knowledge base...
```

Every answer is **grounded**: it is built from the *live aggregated telemetry window*
(SOC, range, battery/coolant temperatures, tyre pressures, powers, DTCs…) plus *retrieved
manual passages* with **citations** `【doc：section】`, and unanswerable questions are
**refused honestly** instead of hallucinated.

A **safety policy layer** classifies every request before it ever reaches the LLM:

| tier | behaviour | example |
|---|---|---|
| `read_only` | routed to the grounded LLM | “What’s my range?” |
| `safe_action` | informational with side-effects | “send a report” |
| `restricted_action` | confirmation required, LLM not consulted | “start charging” |
| `prohibited_action` | **deterministic refusal**, never the LLM | “disable thermal protection”, “reflash ECU” |

Requests that target safety-critical systems (braking, battery management, firmware,
autopilot, speed limits, data exfiltration) are refused deterministically with an
authoritative message — an automotive-grade boundary, not an enterprise chatbot.

## System map

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AURORA (physically-grounded)                     │
│                                                                        │
│  telemetry-simulator ──WS/HTTP──► assistant-api ──RAG──► rag-service    │
│   ·8101  physics-lite EV model     ·8100  intent+grounding  ·8103 BM25 │
│   ·scenarios: leak/dtc/drift/      │         core            KB: 9 docs │
│      charge/battery_heat            ├─context window──► llm-gateway     │
│                                     │                   ·8102 router ───┤
│                                     │        providers: azure-openai ▶ │
│                                     │                    openai ▶     │
│                                     │                    mock (offline)│
│                                     ▼                                 ▼
│  monitoring ·8104  drift (PSI+KS)  eval gate · pass_rate ≥ 0.80 ─ CI   │
│        └── Prometheus /metrics      └── golden set, replay snapshot    │
└────────────────────────────────────────────────────────────────────────┘
```

### Services

| service | port | role |
|---|---|---|
| `telemetry-simulator` | 8101 | physics-lite EV stream (1 s), OBD/DTC injection, WS + REST |
| `llm-gateway` | 8102 | routing, circuit breaker, TTL cache, $ cost ledger, safety |
| `rag-service` | 8103 | deterministic BM25 index over the AURORA manual |
| `assistant-api` | 8100 | intents, safety policy, grounding, prompt builder, web dashboard |
| `monitoring` | 8104 | PSI + KS drift detection vs committed baseline, fleet health, AI-quality |
| `evaluation` | – | golden-set + PEFT/LoRA pipeline gate used by CI |

**No cloud required:** the default provider priority is `mock` — the grounded,
deterministic, offline engine that makes CI and evaluations reproducible.

## Quickstart

```bash
make install     # .venv + shared package (editable)

# containerised stack (Docker Desktop required) — one command:
docker compose up -d --build
open http://localhost:8100            # web dashboard

# or run every service in-process on the host:
make services
make demo                             # end-to-end walkthrough + drift demo

make test            # 23 tests (schemas, telemetry math, RAG, mock, eval metrics)
make lint            # compileall + import smoke
make eval-e2e        # acceptance gate against the golden set (expect ≥ 0.80)
```

### The evaluation gate

The same gate CI blocks on (now **16 cases** including safety/adversarial policy cases):

```bash
cd services/evaluation && ../../.venv/bin/python run.py \
    --assistant http://127.0.0.1:8105 --environment ci
# GATE: ✓ ACCEPTED — ship to production   (pass rate 1.0 ≥ 0.8)
```

```bash
# containerised, reproducible in CI:
docker compose run --rm eval --assistant http://assistant-eval:8100
```

Metrics are **grounded**: `signal_accuracy` (did it quote the live numbers correctly,
within tolerance), `signal_presence`, `doc_grounding` (citations match the manual),
`refusal_honesty` (unanswerable ⇒ refusal), `intent_match`, `text_recall`, and
`policy_compliance` (did the safety policy tier classify the request correctly and
refuse where required). Reports are registered to **MLflow** when a tracking server
is configured.

### PEFT / LoRA fine-tuning pipeline

`make data` builds a **production-shaped training corpus** (real-sounding utterances +
 live-vehicle telemetry snapshots + grounded, cited answers + over-sampled safety
 negatives) curated and version-hashed — see [docs/dataset.md](docs/dataset.md) for the
 schema and curation rules. Optionally ground the per-row telemetry in **real UDS field
 data** (72k km, 7 BEVs, Schreiber et al.) with
 `make data TELEMETRY_DIR=/path/to/electric-vehicle-uds-dataset-main`; it falls back to
 the seeded mock without the download.

For the real LoRA experiment, **AURORA v2** builds a much larger corpus grounded in real
public telemetry (`make data-v2 TUM_DIR=... EPFL_DIR=...`): **TUM EV Usage Data**
(CC-BY-NC-4.0) for driving telemetry resampled into temporally-aligned windows, plus
**EPFL Level-3 Charging** (MIT) for timestamp-aligned charging sessions — kept as two
separate domains and never combined. Every row carries full provenance
(`data_type: real_telemetry`, `qa_type: synthetic_generated`) and splits are assigned by
atomic unit to prevent leakage. See [docs/dataset.md](docs/dataset.md#15) and
[docs/dataset-v2-inspection-report.md](docs/dataset-v2-inspection-report.md).

`make finetune` then prepares the train / validation / adversarial splits and publishes a
LoRA adapter. It runs in two modes:

* `make finetune` — **CI-safe simulation**: distills a grounded adapter from the
  mock provider (no torch/GPU), proves the dataset → LoRA config → eval → registry
  *pipeline*.
* `make finetune-real` — genuine **transformers + peft** (LoRA `r=8, α=16, dropout=0.1`
  on `q_proj`/`v_proj`) when those libs + a base model are configured.
* **Colab (real GPU):** open [`colab/train_lora.ipynb`](colab/train_lora.ipynb), Run-all
  on a free T4, and it produces the same Drop-in `adapter/` + `peft_config.json` that the
  repo's eval consumes.

Both emit `peft_config.json` + an `adapter/` into `artifacts/finetune/` and feed the
same golden-set + MLflow registration path. `make compare` runs
[`services/evaluation/compare.py`](services/evaluation/compare.py) to evaluate the
adapter **head-to-head against the baseline** on the identical deterministic golden set
and blocks on any regression. See [docs/finetuning.md](docs/finetuning.md) for the full
production story.

### LLM provider chain

`azure-openai, openai, mock` (ordered failover, weighted routing, per-provider circuit
breaker, TTL cache, spend ledger). Set in `.env`:

```ini
AURORA_GATEWAY_PROVIDER_PRIORITY=azure-openai,openai,mock
AURORA_GATEWAY_AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/
AURORA_GATEWAY_AZURE_OPENAI_API_KEY=...
AURORA_GATEWAY_OPENAI_API_KEY=...
```

The mock provider keeps every demo/CI/eval run working offline regardless of connectivity.

## Truth & reproducibility

* Telemetry: **deterministic replay snapshot** in `services/evaluation/datasets/telemetry_snapshot.json`
  used by the eval harness; live simulator seeded (`AURORA_TELEMETRY_SEED`). Rich signal
  set incl. pack voltage/current, cabin temp, odometer, GPS, DTCs.
* RAG: chunking + BM25 are pure functions ⇒ **index-hash pinned** (`rag-index-v4`). The
  provenanced owner's-manual corpus is the primary authority, layered under a
  lowest-build-order knowledge-pack for driving-coach content the manual never states.
* Prompts/versions: `aurora-prompt-v9` · `rag-index-v4` · shared `vehicle-ai-shared 2.5.0`.
* Drift baseline: `services/monitoring/baseline.json` (6000 pts/signal across 13 signals,
  seed 42; regenerate with `make baseline`).
* AI-quality metrics: Prometheus counters (grounding, refusals, safety blocks, fallbacks,
  cache, cost, tokens) aggregated at `monitoring:/ai-quality`.

## Deployment

```bash
make build     # docker compose build        (java-free, alpine-slim based)
make up        # run the fleet in containers

kubectl kustomize infra/k8s/overlays/prod | kubectl apply -f -   # AKS
cd infra/terraform && terraform init && terraform apply          # Azure RG/ACR/AKS/KeyVault
```

`.github/workflows/aurora-cicd.yml` runs **lint → test → build & push images → eval gate
→ deploy to AKS** on `main`; the eval gate fails the PR if `pass_rate < 0.80`.

## Repository layout

```
shared/vehicle_ai/        shared contracts & utilities (one source of truth)
services/
  telemetry-simulator/    physics-lite EV model + WS/REST + scenarios
  llm-gateway/            providers, router, safety, cache, cost
  rag-service/            BM25 index + knowledge_base/ (9 AURORA manual docs)
  assistant-api/          core service + dashboard (static/)
  monitoring/             drift detector (PSI+KS) + fleet health
  evaluation/             golden set, metrics, MLflow client, run.py, data/ (dataset pipeline)
  scripts/                gen_baseline.py, demo.py
infra/k8s/                kustomize base + prod overlay (AKS)
infra/terraform/          Azure RG · ACR · AKS · KeyVault · MLflow store
tests/                    pytest suite (schemas, telemetry, RAG, mock, eval)
.github/workflows/        aurora-cicd.yml
```

## Docs

* [docs/architecture.md](docs/architecture.md) — deep dive: contracts, grounding, eval math, drift.
* [docs/dataset.md](docs/dataset.md) — the training corpus: schema, sources, curation, leak-free splits.
* [docs/finetuning.md](docs/finetuning.md) — fine-tuning story: dataset, LoRA, baseline-vs-finetuned gate.
* [docs/demo.md](docs/demo.md) — scripted walkthrough incl. drift injection and gate run.

---

*Platform version 2.5.0 — prompt `aurora-prompt-v9` — RAG index `rag-index-v4` — telemetry schema 1.3.0.*