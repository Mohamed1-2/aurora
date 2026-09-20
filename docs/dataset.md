# AURORA · The training dataset (what we fine-tune on)

This document explains **exactly what data AURORA is fine-tuned on, where it comes
from, and how it is sanitized** — so a reviewer can audit the ML data pipeline, not just
the training loop. It is honest about what is real and what is a stand-in.

---

## 1 · The honest headline

AURORA's fine-tuning corpus is a **production-shaped, synthetic stand-in** for real
assistant traffic. In a real deployment it would be **mined from logged, human-verified
assistant interactions**; because this is an offline portfolio we generate it on the
**exact same record schema** so the *schema, curation rules, leak-free splitting and
validation* — the transferable engineering — are genuine and offline-reproducible.

> The dataset is **not** scraped from the web and is **not** generic EV specs. It is the
> thing a vehicle assistant actually needs: real user questions + a live-vehicle
> telemetry snapshot + a grounded, cited answer.

---

## 1.5 · AURORA v2 — real public telemetry (driving + charging)

AURORA v2 is a **separate, additive corpus** built on top of v1's schema and
pipeline. It keeps the exact `instruction / input / output` + `context` +
`provenance` triple contract the LoRA trainer consumes, but replaces v1's synthetic
telemetry with **real measured data from two public EV datasets used for two distinct
domains that are never combined**:

| dataset | license | domain | coherence |
|---|---|---|---|
| TUM Electric Vehicle Usage Data (Schreiber et al., TU München) | CC-BY-NC-4.0 (non-commercial) | **driving / vehicle telemetry** | `timestamp_aligned` |
| EPFL Level-3 EV Charging (DESL, École Polytechnique Fédérale de Lausanne) | MIT | **charging telemetry** | `timestamp_aligned` |

The definitive inspection report is `docs/dataset-v2-inspection-report.md`. The v2
builder lives in `services/evaluation/data/v2/` and is driven by:

    make data-v2 TUM_DIR=<tum-root> EPFL_DIR=<epfl-root>

### 1.5.1 Integrity model (the part that matters)

*Every* v2 row carries provenance that states exactly where its values came from, and
labels the data honestly across four categories:

1. **Real public telemetry** — drive values from a real TUM driving window; charge
   values from a real EPFL session. `data_type: real_telemetry`.
2. **Real public base model** — the HuggingFace base LM the adapter is trained on
   (not in this repo; chosen in Colab).
3. **Generated Q&A** — the user question and the answer are synthesized from that
   telemetry by manually-written templates. `qa_type: synthetic_generated`. Never
   described as a real customer conversation.
4. **Actual model-training results** — real LoRA losses + evaluation numbers from the
   real run only; no simulated metrics.

TUM driving windows are **temporally aligned** (signals resampled onto a common
`10 s` wall-clock grid from the raw timestamped UDS parquet), so SOC, speed,
temperature, and voltage in one example were measured within the same window — never
spliced from different moments. EPFL charging sessions are **timestamp-aligned by
1-minute measurement**, single-session only. Derived values (e.g. `range_km`) are
explicitly flagged `derived`, never presented as measured.

Signals that do **not** exist in the real data (tire pressure, DTCs, drive mode,
range) are **not fabricated**; they are either excluded from that dataset's examples
or flagged `derived`.

### 1.5.2 Split & duplication control

Splits are assigned **by atomic unit** (vehicle episode / charging session), never by
random row, so the same underlying scenario cannot appear in more than one split. An
automatic leak check in the pipeline (`pipeline.split_by_unit`) enforces this.

### 1.5.3 Corpus size (data-driven, not forced)

Per the inspection report, initial target was ~2 500–4 400; empirically the real data
supports ~**12 000** genuinely distinct, real-telemetry-grounded examples because each
row grounds on a unique real window/session (distinct SOC/speed/temp values), even
where question phrasing reuses a curated template set. The number is what the data
supports — quality and provenance are prioritized over hitting an arbitrary count.

---

## 2 · The schema (one row = one grounded interaction)

Every row in `artifacts/dataset/*.jsonl` is a `DatasetRow` (`services/evaluation/data/schema.py`):

| field | meaning |
|---|---|
| `id` | deterministic hash (reproducible curation) |
| `utterance` | **raw user wording, verbatim** (paraphrases, shorthand, typos) |
| `intent` | one of the 14 assistant intents (range, battery, charging, tire, diagnostics, thermal, energy, maintenance, speed, driving_mode, winter, summary, greeting, out_of_scope) |
| `safety_tier` | `read_only` / `safe_action` / `restricted_action` / `prohibited_action` |
| `refusals` | bool — this interaction is a refusal (safety/out-of-scope) |
| `context` | the live-vehicle `TelemetryContext` at that instant (SOC, range, speed, temps, tyre kPa, DTCs, charging) |
| `anchor_signals` | which live signals the answer is grounded on |
| `grounded_answer` | the **human-verified** answer, built only from `context` |
| `citations` | provenance → manual sections |
| `verified` / `user_rating` | labelling for filtering / preference tuning |
| `split` | train / val / test |

---

## 3 · Where the data comes from

Three channels, mirroring production:

1. **Raw user traffic** — varied real-sounding utterances per intent. The point is
   *variety*: "can I make it to the next town?", "dte?", "how charged is the battery?"
   — not just canonical phrasings.
2. **Grounded, human-verified answers** — derived from the live telemetry, never
   hallucinated, always carrying provenance.
3. **Safety / adversarial negatives** — intentionally over-sampled (they are rare in
   real traffic but critical): attempts to disable battery thermal protection, bypass
   ABS, remove ECU speed limiters, plus out-of-scope questions — each with a
   deterministic refusal.

> In production we additionally mine the **eval-gate failures humans correct** — the
> highest-value slice — plus **preference pairs** from `user_rating` for RLHF-style
> tuning.

**Answer diversity (iteration 2):** each intent now has *multiple distinct grounded
phrasings*, selected per row, plus manual-grounded threshold caveats (thermal warnings
above 42 °C, tyre-pressure notices below 220 kPa, cold-range notes, DTC triage from the
OBD catalog). The model therefore learns to *generate* a grounded answer, not memorise a
single template — while every row is still built strictly from `context` (never
hallucinated) and carries provenance.

### 3a · The telemetry context is optionally grounded in real UDS field data

The `TelemetryContext` snapshot that every answer is grounded on can be drawn from
**real field telemetry** instead of the seeded mock. We ingest the
*Electric Vehicle Usage Data Set for Battery Aging Studies* (Schreiber et al., TU
Munich, CC-BY-NC-4.0): 72 000+ km of **UDS/OBD-II** logs from seven BEVs
(VW ID.3 / CUPRA Born, 58 kWh). We consume its published **JSON histograms**
(`data/json/track_{vehicle}_{value_id}.json`) so the ingestor is **pure-stdlib** —
no pandas/pyarrow — and fully offline.

```
make data TELEMETRY_DIR=/path/to/electric-vehicle-uds-dataset-main
```

`services/evaluation/data/telemetry_source.py` maps real signals into the context, so
each interaction's numbers are real-world distributions, not uniform noise:

| context field | real UDS signal (value_id) | real ranges seen |
|---|---|---|
| `soc_pct` | `hv_soc` (900) | 3–100 % |
| `speed_kph` | `vehicle_speed` (4) | 0–160 km/h |
| `battery_temp_c` | `hv_temp_min` (1208) | 2–43 °C |
| `coolant_temp_c` | `coolant_temp_inverter_inlet` (1269) | 0–48 °C |
| `outside_temp_c` | `ambient_air_temp` (15) | −11…40 °C |
| `range_km` | derived from `hv_soc` | consistent km per charge |

Signals with no field counterpart (tire pressure, DTCs, drive/regen power) keep a
documented physics profile. Omitting `TELEMETRY_DIR` (as CI does) falls back to the
seeded mock, so the pipeline never depends on the download.

> Reference: Schreiber, M., Köning, L., Balke, G., et al. "Lab-to-Field Gap in
> Battery Aging Studies". *eTransportation*, 27, 100518, 2026.
> `https://github.com/TUMFTM/electric-vehicle-uds-dataset`

---

## 4 · Curation rules (the interview gold)

`services/evaluation/data/generate.py` → `curate()` applies the rules a data engineer
runs before a dataset enters a release:

1. **Dedupe near-identical grounded interactions** (same intent + normalized utterance).
2. **Safety/refusal rows are exempt from dedup and never down-sampled** — the safety
   signal is preserved and over-sampled.
3. **Intent-balance** non-safety classes toward a target so "range" doesn't dominate.
4. **Validation** (`validate()`) is a data-quality gate that fails loudly on:
   - empty answers
   - refusal rows that don't actually refuse
   - ungrounded non-refusal rows with no citation (the anti-hallucination check)
   - **PII / VIN sweep** (emails, phone numbers, 17-char VINs)
   - **leak-free splits** (each `vehicle_id` appears in exactly one split)

---

## 5 · Leak-free splitting by vehicle

Splits are assigned **by vehicle block**, never randomly:

```
ORION-X-1000..0999 ──► train
ORION-X-1004        ──► val
ORION-X-1005        ──► test
```

Random row-splitting leaks the same vehicle's state into both train and test and
**inflates** eval scores. Block-splitting by vehicle keeps the evaluation honest — the
test set looks like *a vehicle the model has never seen*.

---

## 6 · Versioning & reproducibility

Each build is **version-hashed** like the RAG index:

```
dataset-517233   (CRC of all row ids; changes iff the data changes)
```

`artifacts/dataset/meta.json` pins the version, split counts, intent distribution and
safety ratio. Because eval and fine-tune consume the identical versioned dataset, our
baseline-vs-finetuned numbers are reproducible — no silent data drift between runs.

---

## 7 · The transferable take-away

> *"I collect from three channels: raw traffic with human ratings, manual-derived
> grounded Q&A, and the eval-gate failures humans correct. Every row carries its
> telemetry context and citations. I dedupe, PII-sweep, balance intents, over-sample the
> safety negatives, and split by vehicle so nothing leaks. The dataset is version-hashed
> so my numbers are reproducible. I never train on a hallucinated output."*

---

## 8 · Files

| path | purpose |
|---|---|
| `services/evaluation/data/schema.py` | canonical `DatasetRow` record + enum types |
| `services/evaluation/data/generate.py` | traffic generation, curation, validation, versioning |
| `services/evaluation/data/telemetry_source.py` | real UDS field-data ingestor (optional `--telemetry-dir`) |
| `services/evaluation/data/v2/` | AURORA v2 pipeline: `tum_driving.py`, `epfl_charging.py`, `scenarios.py`, `pipeline.py`, `build.py` |
| `artifacts/dataset/` | the built, curated, versioned corpus (train/val/test JSONL + meta) |
| `artifacts/dataset/v2/` | the v2 real-telemetry corpus (train/val/test + corpus.jsonl + manifest) |
| `docs/dataset-v2-inspection-report.md` | the v2 data-provenance inspection report |
| `services/evaluation/finetune.py` | consumes the curated corpus → LoRA adapter |
| `tests/test_dataset_pipeline.py` | schema / generation / curation / leak / validation / ingestor tests |
| `tests/test_dataset_pipeline_v2.py` | v2 ingestor / scenario / split / validation tests |
