# AURORA · Fine-tuning a domain LoRA (production story)

This document explains **why** and **how** AURORA fine-tunes its assistant, and how the
result is proven safe to ship. It is written to stand alone — a reviewer should be able
to read it and understand the whole fine-tuning *workflow*, not just the training loop.

---

## 1 · The problem we are solving

The baseline assistant is a **grounded, prompt-driven** system: it never answers from the
LLM's priors, only from the *live telemetry window* and *retrieved manual chunks*. That
gives us high trust but two costs:

1. **Latency / cost** — every answer rebuilds a large grounded prompt and round-trips a
   hosted model.
2. **Domain stiffness** — the "voice" and phrasing live in the prompt, not the weights.

Fine-tuning a small **LoRA adapter** lets us push the domain behaviour *into a small
model's weights* so a single answer can be generated faster and cheaper, while staying on
the same data and being held to the **same deterministic gate**.

> **Key principle:** fine-tuning is a *build-time, calibrated* step — a training run that
> happens in a GPU CI job. It is **never** done per-request and never on the vehicle.
> Runtime stays a small self-hosted model.

---

## 2 · What gets fine-tuned (a LoRA adapter)

We train a low-rank adapter rather than the full model because:

- **Parameter-efficient** — only ~0.1–0.5% of weights are learned, so training fits a
  free Colab T4 GAU and produces a tiny (~10 MB) artifact.
- **Surgical** — we freeze the base and learn small corrections to `q_proj` / `v_proj`
  (the query/value projections), which is enough to shift the model's output style and
  groundedness on this narrow domain.
- **Composable** — the adapter is additive: we can A/B the adapter against the baseline
  without retraining the world, and we can swap adapters per-vehicle-class later.

Config (identical in `finetune.py`, the Colab notebook, and `peft_config.json`):

| hyperparameter | value |
|---|---|
| base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (small, CPU/edge friendly) |
| LoRA rank `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.1 |
| target modules | `["q_proj", "v_proj"]` |
| epochs | 3 |
| batch / grad-accum | 4 / 2 |
| lr | 5e-5 |

---

## 3 · The training data (curated, not scraped)

`make data` builds a **production-shaped corpus** in `artifacts/dataset/` (train/val/test
JSONL + meta), then `finetune.py` converts it into the trainer's `instruction/input/output`
format under `artifacts/finetune/run-1/`:

- **train / val / test** — split **by vehicle block** (no leakage), ~90 curated rows with
  real-sounding utterances, a live-telemetry snapshot per row, and grounded, cited answers.
- **Safety/refusal negatives** — over-sampled; teach the model to *refuse* disabling
  safety systems or answering out-of-scope data-hungry questions.

Each row is an `instruction / input / output` triple, e.g.:

```json
{"instruction": "You are AURORA, an in-vehicle assistant grounded in the ORION-X. Answer only from the provided telemetry. If data is missing, refuse honestly.",
 "input": "Are my tyre pressures okay?",
 "output": "Tyre pressures: 240 kPa front-left, 241 kPa front-right, 240 kPa rear-left, 242 kPa rear-right."}
```

> In production this is mined from **logged, human-verified** assistant traffic, gated so
> we never train on hallucinated outputs. The seed here is the honest, canonical version
> of that.

The per-row **telemetry snapshot** can be grounded in **real UDS field data** (72k+ km from
seven BEVs; Schreiber et al., TU Munich, CC-BY-NC-4.0) instead of the seeded random mock:

```bash
make data TELEMETRY_DIR=/path/to/electric-vehicle-uds-dataset-main
```

`telemetry_source.py` turns published JSON histograms into empirical samplers for SOC,
speed, battery/coolant/ambient temperature — so the numbers a row's answer is grounded on
are real-world distributions. CI runs without the download and falls back to the mock, so
the pipeline stays offline-safe. See [`docs/dataset.md`](./dataset.md#3a) for the full
signal map.

**Characteristics of the enriched corpus** (iteration 2):

* **Answer diversity, not template-memorisation** — each intent now has *multiple distinct
  phrasings* selected per row, so the model learns to *generate* a grounded answer rather
  than reproduce one sentence shape.
* **Manual-grounded reasoning** — answers quote the ORION-X manual (charging 10–80% ≈ 33 min,
  maintain 240 kPa tyres, service at 24 mo/30,000 km, winter range loss 15–30% below −5 °C).
* **Threshold caveats** — a battery above 42 °C triggers a thermal warning; tyre pressure
  below 220 kPa triggers an inflation notice; an ambient temp below 0 °C adds a cold-range
  note. All driven by real telemetry values.
* **DTC triage** — diagnostics answers are grounded in
  `docs/05-diagnostics-and-dtc.md` (P0B36, C0031, B0131, …) instead of a generic "a code is
  present."

### 3.1 · AURORA v2 corpus (real public telemetry — preferred training source)

For the real LoRA experiment (`colab/train_lora.ipynb`), the recommended input is the
**AURORA v2 corpus** built from real public telemetry rather than v1's synthetic rows:

```bash
make data-v2 TUM_DIR=<tum-root> EPFL_DIR=<epfl-root>
```

v2 produces a large, real-telemetry-grounded corpus (`artifacts/dataset/v2/`) where each
row is an `instruction/input/output` triple + `context` + full `provenance`. It is built
by `services/evaluation/data/v2/build.py` from:

- **TUM EV Usage Data** (CC-BY-NC-4.0) — driving telemetry, resampled into
  **temporally aligned 10 s windows** from the raw timestamped UDS parquet
  (`coherence: timestamp_aligned`).
- **EPFL Level-3 Charging** (MIT) — charging telemetry, **timestamp-aligned by 1-minute
  measurement** within single real sessions (`coherence: timestamp_aligned`).

**Honest data-label separation** (kept across all v2 rows and docs):

1. **real public telemetry** (`data_type: real_telemetry`),
2. **real public base model** (HuggingFace, chosen in Colab),
3. **generated Q&A** — synthetic questions/answers grounded on the telemetry
   (`qa_type: synthetic_generated`; never presented as real conversations),
4. **actual model-training results** — measured losses/eval from the real run only.

Splits are assigned **by atomic unit** (vehicle episode / charging session), never by
random row, so no scenario leaks across train/val/test. See
`docs/dataset-v2-inspection-report.md` and `docs/dataset.md#15` for full provenance and
the signal-mapping table.

The Colab notebook (`colab/train_lora.ipynb`) **defaults to the v2 splits**. Cell 5
resolves the dataset (prefers v2, then v1 `run-1`, then local copies), Cell 6 prints a
manifest (train/val counts, TUM-vs-EPFL, driving-vs-charging, safety, unique questions,
coherence + qa_type), Cell 7 runs integrity sanity checks (missing provenance, TUM/EPFL
mixing, train/val atomic-unit overlap, fabricated signals), and Cell 18 stamps the same
provenance summary into `peft_config.json` so the trained artifact is self-describing.

---

## 4 · Training two ways

### 4a · CI-safe simulation (default, `make finetune`)

`finetune.py` in `simulate` mode "trains" a deterministic distilled adapter with **no GPU
/ torch / transformers**. It exists to prove the *pipeline* — dataset → LoRA config →
eval → MLflow — on every CI run, offline and fast. It is conceptually a placeholder for
the real adapter; it mirrors the grounded provider's quality by construction.

### 4b · Real LoRA on a GPU (Colab, `colab/train_lora.ipynb`)

The same config, executed for real. On a T4 this is a ~2–5 min run producing a ~10 MB
adapter. Exact steps with the **real-grounded AURORA v2 corpus** (`artifacts/dataset/v2/`):

1. Upload `artifacts/dataset/v2/train.jsonl` and `val.jsonl` to
   `/content/drive/MyDrive/aurora/v2/` (or drop them beside the notebook — cell 5
   auto-detects v2 first, then v1 `run-1`, then local copies).
2. **Runtime ▸ Run all** on a **T4 GPU**: installs `transformers peft datasets accelerate`,
   prints the **v2 manifest** (cell 6) and passes the **data-integrity sanity checks**
   (cell 7), tokenizes the triples into an instruct chat format, wraps the base
   SMoLM-135M in `LoraConfig(r=8, α=16, q_proj/v_proj)`, trains 3 epochs.
3. Cell 18 saves `./adapter/` + `peft_config.json` (the latter stamped with the v2
   dataset manifest + provenance summary + measured train/eval loss).
4. Download `adapter/` + `peft_config.json` back into `artifacts/finetune/v2/` — kept in a
   clearly separate path from the untouched v1 `artifacts/finetune/run-1/`.
5. **No v1 files are modified**; `artifacts/dataset/` (`dataset-819533`) and
   `artifacts/finetune/run-1/` remain exactly as they were.

The output is **byte-for-byte compatible** with the repo contract, so you can download the
`adapter/` folder straight into `artifacts/finetune/v2/adapter/` and run the gate.

### 4c · Step-1 model evaluation (BASE vs BASE+LoRA) — do this before any serving

Before wiring the adapter into a gateway, prove the **model** improved on its own.
`colab/evaluate_model.ipynb` loads the base model and `base + LoRA` side by side and
scores both on the **v2 held-out `test.jsonl`** (1,117 rows the model never trained on) —
**no FastAPI / gateway / deployment**. Each raw response is scored by
`services/evaluation/model_eval.py`:

* **Rule-based, deterministic** — `signal_accuracy` (measured telemetry numbers the
  reference reports are reproduced within tolerance), `grounding` (mentions the right
  signals), `hallucination_rate` (numbers matching no ground-truth value), `text_recall`/`F1`.
* **LLM-as-judge (optional)** — semantic `answer_quality` / `judge_grounding` via a
  caller-supplied judge callable; absent judge metrics are `None` and excluded from
  aggregates so the deterministic path stays dependency-free.

The notebook prints a Base-vs-LoRA delta table and a single honest **`YES`/`NO`** verdict
(LoRA wins only if it beats-or-ties `signal_accuracy` **and** does not regress
hallucination). This is the first **genuine measured** result. Safety is measured
separately on hand-written adversarial probes (the v2 test set is all benign telemetry).

This is a deliberate *ablation*: only the weights change (no RAG / telemetry / server), so
any measured improvement is attributable to the LoRA alone. Only if the verdict is `YES`
should you proceed to §5's system-level gate.

---

## 5 · Proving it's better — baseline vs fine-tuned gate

The fine-tune is **not** trusted on vibes. `services/evaluation/compare.py` runs the
**identical 16-case golden set** (including the safety/adversarial cases) against two
assistant endpoints — the committed baseline and one serving the adapter — using
**deterministic replay telemetry**, and produces a delta table:

```bash
python services/evaluation/compare.py \
    --baseline http://localhost:8100 --finetuned http://localhost:8105 \
    --method lora-r8-alpha16
```

| metric | baseline | finetuned | Δ |
|--------|----------|-----------|----|
| pass_rate | 1.0 | 1.0 | 0.0 |
| mean.signal_accuracy | 1.0 | 1.0 | 0.0 |
| mean.doc_grounding | 1.0 | … | … |
| mean.text_recall | 0.609 | … | … |
| mean.policy_compliance | 1.0 | … | … |

**The CI contract:** the adapter must **beat or tie** the baseline `pass_rate` and the
groundedness metrics, and must **not** regress `policy_compliance`. Otherwise CI blocks —
the fine-tune can only add value, never make the vehicle assistant worse.

---

## 6 · Suggested interview narrative

> "We keep the runtime assistant deterministic and grounded, and fine-tune is a calibrated
> build-time step. I prepare a curated train/val/adversarial set from logged, verified
> traffic; train a rank-8 LoRA on the Q/V projections (runs on a free Colab T4 and yields
> a ~10 MB adapter); and I never trust it — I run the same 16-case golden set against the
> adapter and the baseline and require the adapter to beat or tie every grounded metric and
> uphold the same policy-compliance. That's the line between 'we fine-tuned a model' and
> 'we shipped a safer, faster, cheaper assistant.'"

---

## 7 · Files

| path | purpose |
|---|---|
| `services/evaluation/data/` | dataset schema, generation, curation, validation |
| `services/evaluation/data/v2/` | real-telemetry v2 corpus builder (TUM driving + EPFL charging) |
| `services/evaluation/finetune.py` | consumes curated corpus → simulate/real LoRA trainer |
| `colab/train_lora.ipynb` | real GPU LoRA training, same artifact contract |
| `services/evaluation/model_eval.py` | raw-model (base vs base+LoRA) hybrid scorer |
| `colab/evaluate_model.ipynb` | Step-1 model-level eval: base vs base+LoRA on the v2 held-out set |
| `services/evaluation/compare.py` | baseline vs fine-tuned delta gate |
| `artifacts/dataset/` | the curated, versioned training corpus (see docs/dataset.md) |
| `artifacts/dataset/v2/` | the real-telemetry v2 corpus (preferred for the real LoRA run) |
| `artifacts/finetune/run-1/` | generated splits + adapter/peft_config |
| `shared/vehicle_ai/schemas.py` | `AssistantRequest`/`EvaluationReport` types |
