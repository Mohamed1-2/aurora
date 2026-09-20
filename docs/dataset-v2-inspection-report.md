# AURORA LoRA v2 — Dataset Inspection Report

**Status:** INSPECTION ONLY — no code changes, no data generated.
**Date:** 2026-09-03
**Branch/scope:** v2 dataset proposal (TUM driving + EPFL charging + manual/RAG + safety)

---

## 1. Executive summary

AURORA v2 grounds the assistant on **two real public EV datasets**, used for two
distinct observation domains that are **never combined into one claimed vehicle
state**:

| Dataset | Domain | License |
|---|---|---|
| TUM Electric Vehicle Usage Data (Schreiber et al., TU Munich) | **Driving / vehicle telemetry** | CC-BY-NC-4.0 (NOT for commercial use) |
| EPFL Level-3 EV Charging (DESL, EPFL) | **Charging telemetry** | MIT |

Both datasets contain real measured data. The key finding of this inspection is a
**material upgrade over the v1 approach**: the TUM repository ships **raw
timestamped UDS parquet logs** in addition to the histogram JSON used in v1. This
enables genuinely **timestamp-aligned driving telemetry** (coherence
`timestamp_aligned`), not just the statistically-grounded per-signal sampling
(`per_signal_distribution`) that v1 used.

This report documents exact provenance, available signals, missingness, usable
records, and a proposed scenario budget + split. **No v2 artifacts are produced
until this plan is approved.**

---

## 2. Dataset A — TUM Electric Vehicle Usage Data (driving)

### 2.1 Location & license

- **Path:** `/Users/mohamedaldofani/Downloads/electric-vehicle-uds-dataset-main/`
- **License:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC-BY-NC-4.0). See `LICENSE`. **NonCommercial only** — attribution required.
  This must be stated prominently in any portfolio/doc and the trained adapter
  must not be distributed commercially.
- **Citation:** Schreiber, M., Köning, L., Balke, G., Abo Gamra, K., Kayl, J.,
  Dietermann, B., Urban, R., Grosu, C., & Lienkamp, M. — *Lab-to-Field Gap in
  Battery Aging Studies: Mismatch of Operating Conditions between Laboratory
  Environments and Real-world Automotive Applications* (TUM, Institute of
  Automotive Technology).

### 2.2 Raw records & fleet

- **Vehicles:** 7 series-production BEVs — 5 CUPRA Born (CUP1–CUP5) + 2 VW ID.3
  Pro Performance (ID1, ID2). Field driving in German public road traffic,
  >72,000 km total, multiple years.
- **Raw UDS logs:** `data/uds_data/{vehicle}.parquet` (long format).
- **Pre-processed histograms:** `data/json/` — 286 files (165 `track_*`, 121
  `charging_*`).

### 2.3 Parquet schema (the primary v2 driving source)

| column | type | example |
|---|---|---|
| `vehicle_id` | string | `CUP1` |
| `time` | timestamp[ns] | `2023-04-01 19:17:14.092` |
| `value_id` | int64 | `900` |
| `value` | double | `52.4` |

**Long format:** each row is one (time, signal-value) measurement. Signals are
sampled on **different periodic grids** (voltage 200 ms, SOC 5 s, battery temp
10 s), so cross-signal simultaneity requires **temporal binning** (see §2.7).

### 2.4 Record counts per vehicle (usable 2022–2023)

| vehicle | parquet rows | usable 2022–23 | signals present |
|---|---|---|---|
| CUP1 | 8,659,492 | 8,658,680 (100%) | 12 |
| CUP2 | 3,743,488 | 3,743,488 (100%) | 12 |
| CUP3 | 986,864 | 986,864 (100%) | 12 |
| CUP4 | 4,943,471 | 4,943,471 (100%) | 12 |
| CUP5 | 3,305,981 | 3,305,981 (100%) | 12 |
| ID1 | 40,401,700 | 16,242,800 (40%) | 15 |
| ID2 | 36,020,218 | 18,375,620 (51%) | 12 |
| **Total** | **~98M** | **~58M usable** | — |

Notes:
- ID1/ID2 include 2021 data that falls outside the aligned 2022–23 window used
  for consistency; ~50% retained.
- CUP1 contains a small number of corrupted future timestamps (e.g. year 2087);
  these are excluded during cleaning.

### 2.5 Available signals (value_id → name) in parquet

Present in **all 7 vehicles** (12 core signals):

| value_id | variable_name | unit | sampling (from value_overview.csv) |
|---|---|---|---|
| 4 | vehicle_speed | km/h | 200 ms |
| 15 | ambient_air_temp | °C | 10 s |
| 56 | hv_aux_power | W | 1 s |
| 900 | hv_soc | % | 5 s |
| 961 | temp_rear_motor_stator | °C | 5 s |
| 1200 | hv_battery_voltage | V | 200 ms |
| 1208 | hv_temp_min | °C | 10 s |
| 1209 | hv_temp_max | °C | 10 s |
| 1265 | temp_rotor_rear_motor | °C | 5 s |
| 1269 | coolant_temp_inverter_inlet | °C | 5 s |
| 1272 | hv_battery_temp_inlet | °C | 10 s |
| 1273 | hv_battery_temp_outlet | °C | 10 s |

Present only in **ID1** (3 signals):

| value_id | variable_name |
|---|---|
| 43 | interior_temp |
| 1205 | ptc1_current |
| 1207 | ptc_voltage |

Only available as trip-level **summary histograms** (not per-measurement in the
aligned driving series): DOD (1290), track_duration (1291), cell voltage
max/min/delta (1293/1294/1295), cell c-rate peaks (1302/1303), traveled distance
(1299). These are per-track statistics, not continuous timed series.

### 2.6 Signals that DO NOT exist in TUM (must not be fabricated as real)

The following are **absent** from the driving parquet and must NOT be presented
as real driving telemetry:
- **tire pressure** (any wheel)
- **DTC / OBD trouble codes**
- **drive mode** (Eco/Sport/Comfort)
- **cabin HVAC setpoint** (interior temp exists only on ID1, as measured value)
- **range / distance-to-empty** (km) — only `traveled_distance` summary exists;
  range is a **derived** quantity
- **regen power (kW)** — not in TUM driving parquet

### 2.7 Feasible driving scenarios & coherence model

**Coherence model: `timestamp_aligned` via temporal binning.** Because signals
share a real wall-clock `time`, we can resample all present signals into a common
window grid (e.g. 5 s or 10 s windows) and, for each window, hold the value **from
the same vehicle within that window** (nearest-sample). This yields a genuinely
simultaneous multi-signal vehicle state per window — the exact "don't splice A/B/C"
integrity property required. This is the **primary v2 coherence choice** and
supersedes v1's `per_signal_distribution` for driving.

- Only windows where all selected signals have a value within tolerance are kept.
- Only signals present for that vehicle are used (no ID1-only signals on CUPx).
- Corrupted timestamps (2087 outliers) and 2021-only spans are excluded.

**Scenario derivation:** contiguous valid windows are grouped into *driving
episodes*; each episode yields one or more instruction/context/output examples
(SO-level low-SOC, thermal, speed+SOC, voltage, etc.). Episodes, not rows, are the
atomic unit for train/val/test splitting (leakage control, §6).

### 2.8 Histogram JSON (v1 source — retained, secondary)

`data/json/track_*.json` = per-vehicle per-signal **1D histograms**
(`x_values` = bins, `y_values` = density). These are trip-aggregated and therefore
*not* cross-signal simultaneous. They remain the basis of v1's
`telemetry_source.py` (coherence `per_signal_distribution`) and are **kept
unchanged** for the v1 baseline/harness. v2 driving uses the parquet series as the
authoritative source; the histogram path is retired for v2 driving coherence but
left intact.

---

## 3. Dataset B — EPFL Level-3 EV Charging

### 3.1 Location & license

- **Path:** `/Users/mohamedaldofani/Downloads/Level-3-EV-charging-dataset-main/`
- **License:** MIT (Copyright (c) 2025 Distributed Electrical Systems Laboratory
  (DESL), École Polytechnique Fédérale De Lausanne).
- **Files:** `Measurement_data.xlsx`, `Session_data.xlsx`, `LICENSE`, `README.md`.

### 3.2 Collection context

Level-3 DC fast-charging station, southwestern Switzerland; measured
2022-04-12 → 2023-07-04. 6 plugs (only CCS1/CCS2 included). Max station power
172.5 kW; two EVs can charge simultaneously with shared capacity.

### 3.3 Measurement dataset (timestamp-aligned, 1-min resolution)

| identifier | description | unit |
|---|---|---|
| Date and time | measurement timestamp | — |
| CCS | plug (CCS1/CCS2) | — |
| Session | station session id | — |
| P | measured charging power | W |
| Pset | power setpoint | W |
| Preq | requested power | W |
| TotalCapacity | total battery energy capacity | Wh |
| RemainingCapacity | remaining energy | Wh |
| BulkCapacity | usable battery energy capacity | Wh |
| Energy | cumulative charged energy | Wh |
| SOC | state of charge | % |
| Ctrl_t / Ctrl_s | controlled-charging booleans | 0/1 |

- **Total rows:** 646,680. **Rows with actual data:** 36,184 (5.6%) — the rest are
  **empty placeholders** (no SOC/power). Usable rows: **36,184**.
- **Usable sessions (with measurements):** **1,129** (Session ids 1..1129).
- **Mean ~32 measurement rows per usable session.**
- **Unique plugs:** `CCS1` only in the measurement data.
- **Coherence:** `timestamp_aligned` — SOC, P, Preq, Pset, Energy are measured on
  the same 1-minute grid within a session. **Measured within a single real
  session only.**

### 3.4 Session dataset (1878 sessions, all fields complete)

| identifier | description | unit |
|---|---|---|
| Session | id (1..1878) | — |
| CCS | CCS1/CCS2 | — |
| Arrival / Departure | timestamps | — |
| Stay | session duration | min |
| Energy | charged energy | Wh |
| Pmax / Preq_max | max power | W |
| Controlled session | Pset<Preq anytime | 0/1 |
| TotalCapacity / BulkCapacity | battery capacity | Wh |
| SOC arrival / SOC departure | SOC at start/end | % |
| Energy capacity | approximated capacity | Wh |

- **Total sessions:** 1,878 — **all fields complete** (1878/1878 non-null).
- SOC arrival 0–98%, departure 12–100%. Stay 5–144 min (mean 32.9).
- **75% of sessions are "controlled"** (station limited power below request) — an
  excellent real-world edge case for "why is charging slower than expected".

> **Note on the earlier README summary vs actual data:** the README states
> "64,277 measurements" and "1,878 sessions". The raw workbook contains
> 646,680 rows but only **36,184 carry values** (5.6%); usable sessions with
> measurements = **1,129**. We report the *measured/usable* counts, not the
> workbook's empty-row count.

### 3.5 Signals that DO NOT exist in EPFL

- **Ambient temperature, battery temperature, coolant temperature** — not in the
  charging data. Charging examples must not report these as measured.
- **Location / vehicle model** — not provided (obfuscated).
- **Simultaneous power across both plugs** is implicit (shared station capacity)
  but is not reconstructed here.

---

## 4. Signal mapping (TUM driving, v2)

Canonical AURORA context field ← TUM parquet signal. **Only derived fields are
flagged as derived.**

| AURORA context | TUM source (value_id) | type | notes |
|---|---|---|---|
| `soc_pct` | hv_soc (900) | measured | |
| `speed_kph` | vehicle_speed (4) | measured | |
| `battery_temp_c` | hv_temp_max (1209) | measured | use max for worst-case thermal |
| `battery_temp_min_c` | hv_temp_min (1208) | measured | |
| `coolant_temp_c` | coolant_temp_inverter_inlet (1269) | measured | |
| `outside_temp_c` | ambient_air_temp (15) | measured | |
| `pack_voltage_v` | hv_battery_voltage (1200) | measured | |
| `motor_stator_temp_c` | temp_rear_motor_stator (961) | measured | |
| `hv_aux_power_w` | hv_aux_power (56) | measured | |
| `range_km` | — | **derived** | estimated from SOC × usable capacity; explicitly `derived:true`, not measured |
| `tire_pressure_*` | — | **unavailable** | excluded from TUM driving examples |
| `dtc_codes` | — | **unavailable** | excluded / safety-only narrative |
| `driving_mode` | — | **unavailable** | excluded as measured; may appear only as user intent, not telemetry |
| `interior_temp_c` | interior_temp (43) | measured | **ID1 only**; excluded from CUPx |

**EPFL charging context** (all measured within a session):
`soc_arrival_pct`, `soc_departure_pct`, `soc_pct` (mid-session), `charging_power_kw`
(P/1000), `power_setpoint_kw` (Pset/1000), `requested_power_kw` (Preq/1000),
`energy_charged_kwh` (Energy/1000), `session_duration_min` (Stay),
`total_capacity_kwh` (TotalCapacity/1000), `bulk_capacity_kwh` (BulkCapacity/1000),
`controlled_session` (bool).

Unit conversions (W→kW, Wh→kWh) are stated as transformations.

---

## 5. Missingness & data-quality summary

| aspect | TUM driving (parquet) | EPFL charging |
|---|---|---|
| timestamps available | YES (ns, wall-clock) | YES (1-min) |
| cross-signal simultaneity | achievable via binning | inherent (same grid) |
| row-level missingness | sparse-value long format; cleaned per signal | 94% empty placeholders dropped |
| corrupted timestamps | few (2087 outlier) — excluded | none found |
| usable records | ~58M (2022–23) | 36,184 measurements / 1,129 sessions |
| per-session atomic unit | driving episode | charging session |

---

## 6. Proposed corpus budget & split

Per decision #8, **no forced 20k target**; quality/provenance first. Realistic
distinct scenario budget from actual source capacity:

| source | atomic units available | proposed examples |
|---|---|---|
| TUM driving (aligned windows → episodes) | ~7 vehicles, ~58M rows → thousands of distinct episodes | ~1,200–2,000 |
| EPFL charging | 1,129 real sessions (+ 1,878 session summaries) | ~800–1,500 |
| Manual/RAG (ORION-X owner's manual KB) | existing docs | ~300–500 |
| Safety / refusal / edge | curated negatives | ~200–400 |
| **v2 total** | | **~2,500–4,400** |

**Split strategy (leakage control — decision #9):** split by **atomic unit
(vehicle-episode / charging-session / manual-section)**, NOT random rows, so no
underlying telemetry scenario appears in more than one split.

- Train ~75–80%, Val ~10–12%, Test ~10–12%.
- **TUM:** hold out entire vehicles or non-overlapping episodes for val/test so a
  scenario seen in train (same vehicle+time) is never re-shown worded differently
  in val/test.
- **EPFL:** split by **session id** (never splinter a session across splits).
- Deterministic seed; record split assignment + hashes for reproducibility.

---

## 7. Provenance schema (per decision #6)

Every row carries:

```jsonc
{
  "_id": "TUM-DR-000123",
  "instruction": "My battery is at 17%. Is that low?",
  "input": "My battery is at 17%. Is that low?",
  "output": "Your battery is at 17.4% SOC, which is relatively low...",   // synthetic answer, grounded
  "context": { "soc_pct": 17.4, "speed_kph": 83.0, "battery_temp_c": 34.2, "outside_temp_c": 2.0 },
  "intent": "battery",
  "safety_tier": "normal",
  "provenance": {
    "source_dataset": "TUM_UDS",
    "source_record": "data/uds_data/CUP1.parquet",   // exact file(s)
    "state": "driving",
    "vehicle_id": "CUP1",
    "window_start_utc": "2023-04-01T19:17:14",
    "window_duration_s": 10,
    "episode_id": "CUP1-20230401-ep1",
    "signals_used": ["hv_soc","vehicle_speed","hv_temp_max","ambient_air_temp"],
    "coherence": "timestamp_aligned",
    "state_reconstruction": "temporal_binned_window",
    "data_type": "real_telemetry",
    "qa_type": "synthetic_generated",
    "answer_grounding": "synthetic",
    "human_verified": false
  }
}
```

For EPFL examples:
```jsonc
"source_dataset": "EPFL_L3_CHARGING",
"state": "charging",
"session_id": "42",
"coherence": "timestamp_aligned",
"state_reconstruction": "measured_session"
```

Classification labels guaranteed per row:
- `data_type` ∈ {`real_telemetry`}
- `qa_type` ∈ {`synthetic_generated`, `curated_answer`, `human_verified`}
- `human_verified` ∈ {true, false}

**Honesty rule:** synthetic user questions + generated answers are NEVER described
as real customer conversations.

---

## 8. Documentation separation commitment (decision #7/#8 + portfolio)

The docs will keep four clearly-boxed categories:
1. **Real public telemetry** (TUM measured driving, EPFL measured charging).
2. **Real public base model** (SnagLM2-135M-Instruct or chosen base), Apache-2.0.
3. **Generated Q&A** (synthetic instructions/answers grounded on 1–2, always
   labeled).
4. **Actual model-training results** (real LoRA losses + eval numbers from the
   real run only; no simulated metrics).

---

## 9. Proposed next phase (blocked on approval)

Only after this report is approved:

1. **Ingestor `tum_driving.py`** — read parquet long-format, bin into aligned
   windows, drop corrupted/out-of-range, emit `TUMDrivingWindow` records with
   provenance.
2. **Ingestor `epfl_charging.py`** — read workbook, drop empty rows, expose
   aligned sessions + session summaries with provenance.
3. **Scenario builder** — transform real windows/sessions → instruction/context/
   output triples via parametrized, manually-written templates (driving:
   low-SOC, thermal, speed+SOC, voltage, cold-range; charging: energy, duration,
   power lower than expected, SOC at start/end, session summary, controlled/
   unusual situations; + manual/RAG; + safety).
4. **Merge, dedup, split-by-unit, validate** (schema, provenance, coherence,
   leak, PII).
5. v2 artifact under `artifacts/dataset/v2/` — **leaves v1
   `dataset-819533` and `artifacts/finetune/run-1` untouched.**

v1 dataset and v1 harness are preserved as the baseline. v2 is a separate,
additive pipeline.

---

*End of inspection report. Awaiting approval before any v2 code or data is
generated.*
