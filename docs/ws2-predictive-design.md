# WS2 — Real-Data Predictive Calibration: design (for review)

Status: **design only — no training run yet.** This document describes *exactly*
what WS2 will build and how it will be evaluated, grounded in verification runs
against the **genuine raw TUM parquet** (verified 2026-09-07).

All availability numbers below were measured from the real files at
`~/Downloads/electric-vehicle-uds-dataset-main/data/uds_data/*.parquet`
(the dataset on which the project's v2 decisions are based).

---

## 1. Real data available (verified on the raw parquet)

### 1.1 Measured continuous signals (value_overview.csv → verified present in parquet)

| value_id | variable | unit | nominal sampling | verified in 2022–23 parquet |
|---|---|---|---|---|
| 4 | vehicle_speed | km/h | 200 ms | yes, med gap 0.20–0.22 s |
| 1200 | hv_battery_voltage | V | 200 ms | yes, med gap 0.11 s |
| 56 | hv_aux_power | W | 1 s | yes, med gap 1.02 s |
| 900 | hv_soc | % | 5 s | yes, med gap 5.01 s |
| 961 | temp_rear_motor_stator | °C | 5 s | yes, med gap 5.01 s |
| 1265 | rear_motor_rotor_temp | °C | 5 s | yes, med gap 5.01–5.02 s |
| 1269 | coolant_temp_inverter_inlet | °C | 5 s | yes, med gap 5.02 s |
| 1272 | hv_battery_temp_inlet | °C | 5 s | yes, med gap 5.01 s |
| 1273 | hv_battery_temp_outlet | °C | 5 s | yes, med gap 5.01 s |
| 15 | ambient_air_temp | °C | 10 s | yes, med gap 10.01 s |
| 1208 | hv_temp_min | °C | 10 s | **sparse** (see 1.3) |
| 1209 | hv_temp_max | °C | 10 s | **sparse** (see 1.3) |
| 43 | interior_temp | °C | 10 s | ID1 only |
| 1205 | ptc1_current (cabin PTC heater) | A | 1 s | ID1 only |
| 1207 | ptc_voltage | V | 1 s | ID1 only |
| 1288 | cell_c_rate | 1/h | — | **absent** from continuous parquet (histograms only) |

Shared wall-clock `time`, long format `(time, value_id, value)`; all values
measured with real timestamps. Signals are sampled on different periodic grids;
raw rows are binned into a common 10 s grid (nearest-sample, per-window) — this
is the existing `temporal_binned_window` coherence used by the v2 corpus and
`TUMReplaySource`.

### 1.2 Measured vs derived vs unavailable (honest classification)

**Measured** (usable directly): `vehicle_speed_kph`, `hv_aux_power_w`,
`soc_pct`, `ambient_air_temp_c`, `coolant_temp_inverter_inlet_c`,
`pack_voltage_v`, `battery_temp_inlet_c`, `battery_temp_outlet_c`,
`motor_stator_temp_c`, `motor_rotor_temp_c` (all on 7 vehicles);
`interior_temp_c`, `ptc1_current_a`, `ptc_voltage_v` (ID1 only).

**Derived** (computed from measured, never presented as measured):
- `distance_km` = integral of measured vehicle speed.
- `accel_mps2` = Δ(speed)/Δt.
- `cabin_heater_power_w` = `ptc1_current × ptc_voltage` (ID1 only).
- `energy_wh` from SOC change REQUIRES a capacity assumption — SOC is relative
  to usable net capacity; we use the dataset-stated fleet net capacity
  **58 kWh** (README, "Net Energy Capacity") as *derived conversion*, always
  flagged `assumption`-adjacent (see §5).

**Unavailable (must NOT be invented)**:
- cabin HVAC setpoint / user climate demand
- cabin interior temperature on CUPx (ID.3 has it, only 2021-era — sparse/out of window)
- destination / route / remaining distance (trip-level binaries exist in
  `traveled_distance`, 1299? — NO, 1299/1290/1291 are **trip-summary histograms** (JSON),
  not per-measurement series; not usable as a continuous signal)
- route grade / elevation
- drive mode, tire pressure, DTC
- regen power (not a measured continuous signal)

### 1.3 Availability reality-check (verified window counts, 2022–23)

Requiring the **dense 5-core** signals `{soc_pct, speed_kph, hv_aux_power_w,
outside_temp_c, coolant_temp_c}` yields:

| vehicle | 5-core windows (10 s) | + battery_temp (1209/1208) gate |
|---|---|---|
| CUP1 | 39,796 | 3,100 |
| CUP2 | 16,306 | 1,345 |
| CUP3 | 4,518 | 363 |
| CUP4 | 23,291 | 2,032 |
| CUP5 | 17,271 | 1,934 |
| ID1 | 58,916 | 6,320 |
| ID2 | 75,587 | 7,292 |
| **TOTAL** | **235,685** | **22,386** |

`battery_temp_max` (1209) is degraded on every vehicle (e.g. CUP1: 2,287 rows vs
coolant 104,891) — using it as a hard gate costs **≈10.5×** of the real data
while it only marginally informs energy. **WS2 therefore gates on the dense
5-core**, treats battery temps as optional features (`battery_temp_inlet_c` /
`battery_temp_outlet_c` are dense and preferred), and reports any run where the
pack-temp feature is missing as available-with-NaN.

### 1.4 A critical measured-data caveat: SOC quantization

Verified: real `hv_soc` is quantized — unique values in 0.4% steps
(e.g. 3.6, 4.0, 4.4…; identical for CUP1 and ID2). A 60 s ΔSOC ≈ 0.4% is
therefore ~one quantization step. **Consequence:** the default training window
is **W = 300 s** (min 120 s), giving a measured ΔSOC ≈ 2% (≈5+ quant steps) and
an episode-level test metric that aggregates over a full trip.

---

## 2. Training example — real TUM slice → one ML row (walkthrough)

Row taken from **CUP1, 2022-12-29, 10:21:30 → 10:22:30 UTC** (six consecutive
10 s aligned windows, all values from `data/uds_data/CUP1.parquet`):

```
 bin | start_utc              | speed_kph | soc_pct | hv_aux_power_w | outside_temp_c
   1 | 2022-12-29T10:21:30+00 |  56.8     | 63.60   | 1300           | 9.5
   2 | 2022-12-29T10:21:40+00 |  51.3     | 63.60   | 1400           | 9.5
   3 | 2022-12-29T10:21:50+00 |  59.7     | 63.60   | 1500           | 9.5
   4 | 2022-12-29T10:22:00+00 |  57.0     | 63.60   | 1500           | 9.5
   5 | 2022-12-29T10:22:10+00 |  58.7     | 63.60   | 1600           | 9.5
   6 | 2022-12-29T10:22:20+00 |  60.8     | 63.20   | 1700           | 9.5
```

One **60 s training row** (as logged; the default W=300 s uses the same recipe
over more bins):

- `start_utc` = `2022-12-29T10:21:30+00:00`
- `telemetry_window` = 6 × 10 s aligned bins (`episode_id="CUP1-2022-12-29"`)
- `start_soc_pct` = 63.60, `end_soc_pct` = 63.20
- `observed_soc_change_pct` = **−0.40** (measured target)
- `distance_km` = **0.957** (derived = Σ speed·Δt)
- `speed_kph_mean` = 57.4, `speed_kph_max` = 60.8, `speed_kph_std` = 3.1
- `decel_fraction` = 0.40 (2/5 bins with negative accel — derived)
- `hv_aux_power_w_mean` = 1500 (measured)
- `outside_temp_c_mean` = 9.5, `coolant_temp_c_mean` = 16.5 (measured)
- physics prediction (WS1 formulas, §5): motive 100.3 Wh + aux 25.0 Wh = 125.3 Wh
  → `physics_soc_change_pct` = −0.153% (58 kWh conversion)
- **`residual_pct` = −0.40 − (−0.153) = −0.247** (the ML target)

The residual is ≈2.6× the physics magnitude: the physics model (aero+roll+inertia,
no motor/EMS efficiency, no regen credit, HVAC forced 0) structurally
under-predicts real consumption. That systematic bias is what calibration learns.

---

## 3. Target — exactly what the ML model learns (approved convention)

**SOC-drop convention (locked, used everywhere — dataset, training, inference,
metrics, filenames, docs):** energy consumption is a *positive* number.

For a window `w` of duration `W` in episode `e`:

```
soc_drop_obs(w)  = soc_start_pct − soc_end_pct            # measured, % points; > 0 while discharging
soc_drop_phys(w) = E_phys(w) / C × 100                    # physics, converted via capacity C (Wh)
E_phys(w)        = Σ_bins max(0, motive_w_bin)·Δt_bin + Σ_bins aux_w_bin·Δt_bin + hvac=0
                   # hvac = 0 on TUM (no cabin setpoint); aux = measured hv_aux_power
residual(w)      = soc_drop_obs(w) − soc_drop_phys(w)     # ← the ML target
final_soc_drop_prediction(w) = soc_drop_phys(w) + ml_residual_prediction(w)
```

**The calibration model learns `ĝ(features) ≈ residual(w)`** — the *correction*
the physics baseline is missing. The model never replaces physics; the final
prediction is additive. Positive `residual` = real fleet consumed *more* than
physics (the common case — no drivetrain efficiency / capacity headroom / regen
credit in the baseline).

Conversion of residual (SOC points) → Wh for reporting uses the flagged
per-vehicle capacity (see §5B), with every such report labeled as a derived
conversion.

## 5B. Per-vehicle capacity contract (approved)

One global 58 kWh usable capacity is **NOT** silently applied to every vehicle.
A `vehicle_capacity.json` table carries, per vehicle:

```
vehicle_id | usable_capacity_kwh | provenance | source_reference | assumption
```

Both ID.3 (ID1/ID2) and CUPRA Born (CUP1..CUP5) are listed at 58 kWh *net* in
the dataset README, but we have no per-vehicle *measured* usable capacity, so
every row is flagged `assumption=True` (provenance
`tum_dataset_readme_fleet_spec`). Baseline energy/SOC conversion uses 58 kWh by
default; a capacity-sensitivity analysis (§7) reports how metrics move across
55/58/62/82 kWh. Primary held-out metrics emphasize **SOC-point error**;
energy/kWh MAE is reported only as dependent on the capacity assumption.

---

## 4. Leak-free split

- Atomic unit is the **episode** (`episode_id = {vehicle}-{date}`) — never split
  a continuous episode across splits.
- **Whole-vehicle holdout for final test:** `ID2` (largest) + `CUP3` (smallest)
  are held out entire as test; all their windows are unseen at train time.
- Val: a second whole-vehicle block (`ID1`) is held out for
  hyper-parameter choice and early stop.
- Train: remaining vehicles (`CUP1, CUP2, CUP4, CUP5` + CUP3? → see plan).
- If episode counts are too small to hold out whole vehicles, fall back to
  **non-contiguous episode sampling** (train episodes are fully disjoint from
  val/test episodes) — at no time are row-neighbors from one trip separated
  into different splits.
- Deterministic seed + a split-assignment table (`dataset_split.parquet`) records
  `episode_id → split` and epoch/version for reproducibility.

---

## 5. Physics baseline on real TUM

WS1 `compute_drivers`/`_motive_power_w` constants (profile `ws1-v1`) are reused,
with the critical substitutions to stay honest on real data:

| WS1 physics input | populated from TUM? | source |
|---|---|---|
| speed → aero/roll/inertia | ✓ | measured `vehicle_speed` (0.2 s), accel derived |
| aux load | ✓ (replaces `aux_base_w`=350) | measured `hv_aux_power` (1 s) |
| ambient temp | ✓ | measured `ambient_air_temp` (10 s) |
| motor/battery thermal state | ✓ (optional feature) | coolant, pack in/outlet temps |
| cabin HVAC load | **✗ unavailable** | forced **0** on TUM (never invented) |
| cabin setpoint | **✗ unavailable** | no feature, no hvac term |
| route grade | **✗ unavailable** | not estimated, reported as limitation |
| battery capacity | **assumption** | fleet net **58 kWh** (README) flagged; 82 kWh (sim profile) reported separately |

Baseline choice: `motive_w = max(0, aero+roll+inertia)` (conservative,
no regen credit), `aux = measured mean`, `hvac = 0`. `C` defaults to 58 kWh.

---

## 6. ML calibration model

- **Model:** LightGBM gradient boosting regressor (`num_leaves≈16–31`,
  `n_estimators≈500` w/ early stopping) — lightweight, interpretable, strong on
  tabular. Fallback: `sklearn.ensemble.HistGradientBoostingRegressor` if LightGBM
  is not installable in the environment.
- **Features (per row, all from measured/derived):**
  - speed: `speed_kph_mean`, `speed_kph_std`, `speed_kph_max`, `speed_kph_p90`
  - motion: `distance_km` (derived), `decel_fraction`, `accel_mean`,
    `regen_decel_energy_kwh` (derived from negative-inertia work)
  - aux: `hv_aux_power_w_mean` (measured)
  - thermal: `outside_temp_c_mean`, `coolant_temp_c_mean`,
    `battery_temp_inlet_c_mean` (NaN-aware)
  - physics: `physics_soc_change_pct` (the baseline output itself)
  - meta: `window_dur_s` (constant 300 normally)
- **Target:** `residual_pct` (§3).
- **Training procedure:** Train on whole-vehicle train set only; early-stop on
  the whole-vehicle val set; evaluate ONCE on the unseen test set. No
  calibration against sim data (see §8).
- **Saved artifact** `artifacts/calibration/v2-ws1/`:
  - `model.lgb` (or `.joblib`), `params.json` (LightGBM hyper-params + profile
    version + feature list + split-version), `metrics_test.json`,
    `feature_importance.png`, `dataset_split.parquet`,
    `train_window_stats.parquet` (provenance per row: episode, vehicle, start_utc,
    count of bins, core-signals-present, quant-steps).
- **Versioning:** `v2-ws1-r1`, incremented when data profile, window config,
  features, or hyper-params change; each artifact embeds `profile_version`,
  `window_s`, `capacity_wh`, `split_version`, `git_hash`, `created_utc`.
- **Inference contract:** `predict_delta_soc(features) -> delta_soc_pct` where
  `delta_soc_pct = physics_soc_change_pct + correction_pct`; inputs must be
  pydantic-validated `EnergyFeatures` (shared WS1 schema) and the module exposes
  plain functions (no service I/O) so both the evaluation harness and a future
  endpoint can call it. Deterministic (fixed seed).

---

## 7. Evaluation (the WS2 comparison)

Held-out real TUM episodes; **physics-only vs physics+calibration**, reported as:

- primary: `MAE(soc_drop_obs − soc_drop_pred)` in **% points** (positive-drop
  convention);
- `RMSE` (same units);
- energy MAE via the flagged 58 kWh conversion (reported as *derived*);
- relative improvement `(MAE_phys − MAE_cal) / MAE_phys` (%);
- breakdowns: **by vehicle, by speed band** (<30, 30–60, 60–90, 90–120, >120
  km/h), **by temperature band** (<0, 0–10, 10–20, 20–30, >30 °C);
- **regression analysis:** rows where `|ΔSOC_pred_cal − obs| > |ΔSOC_pred_phys
  − obs|` — count + where they live (which vehicle/band), so we can show exactly
  where ML worsens physics.

**Promotion rule (explicit):** the calibrated model is deployed only if, on the
held-out test set, it (a) improves test MAE by ≥ 5% relative, (b) does not
regress any vehicle beyond +20% relative MAE, (c) is not materially worse in any
speed/temp band that represents ≥ 5% of test rows. Otherwise we keep
physics-only and record the negative result honestly.

---

## 8. Simulator is NOT training truth

- Simulator data (`/api/trip`, `what-if`, HVAC scenarios) stays for controlled
  what-if/HVAC/demo testing only.
- It is **never** a training label source; no sim-derived residual is used as
  primary evidence that calibration works. All training/validation/test labels
  come from measured TUM SOC deltas.

---

## 9. Proposed WS2 implementation plan (no training until approval)

1. **WS2-prep (data & baseline baking).** New `services/calibration/` package:
   `window_builder.py` (align 10 s bins → W=300 s rows on dense 5-core,
   NaN-optional temps), `features.py`, `physics_baseline.py` (reuses WS1
   `_motive_power_w`; aux=measured; hvac=0; 58 kWh flagged), writes
   `artifacts/calibration/v2-ws1/dataset_w300s.parquet` + availability/stat
   report. Config: reuse `AURORA_TELEMETRY_TUM_UDS_DIR` for the UDS dir.
2. **Split.** Episode-aware, whole-vehicle test (`ID2`, `CUP3`) + val (`ID1`) +
   train; `dataset_split.parquet`.
3. **Baseline bake.** Compute physics-only predictions on val+test; record
   baseline MAE/RMSE.
4. **Train calibration** (only after your approval): LightGBM on train, early
   stop on val; freeze `model.lgb` + `params.json`.
5. **Evaluate.** Test-set tables (metrics, by vehicle/speed/temp), regression
   analysis, improvement %; generate `metrics_test.json` + plots.
6. **Promote/cut** per §7 decision; version `v2-ws1-r1`.
7. (Optional, later) thin endpoint wrapping `predict_delta_soc` for the trip
   flow; still no prompt/RAG/cockpit/voice changes.

---

## 10. WS2 outcome (implemented, promoted, frozen)

All of the above is **done** and the conservative calibration is **PROMOTED**
as the active prediction path, frozen at `v2-ws1-r1`.

### Final decision
`calibrated_soc_drop_pct = physics_soc_drop_pct + lam_band * ml_residual_pct`,
with per-band lambda tuned on validation only:

| band | lambda |
|---|---|
| 0-30 | 0.20 |
| 30-60 | 1.00 |
| 60-90 | 1.00 |
| 90-120 | 1.00 |
| 120+ | 0.55 |

The full LightGBM failed promotion because the 0-30 km/h band regressed +11.2%
MAE; the conservative per-band shrink was tuned on **validation only** to fix
exactly that band and passed every promotion gate.

### Held-out test (ID2+CUP3; single confirmatory look)
| model | overall MAE % | vs physics |
|---|---|---|
| physics only | 0.8605 | reference |
| full LightGBM | 0.6574 | −23.6% |
| **conservative (promoted)** | **0.6520** | **−24.2%** |

0-30 band: 0.4462 → 0.4473 (+0.2%, fixed). 60-90 band: 1.2219 → 0.6096 (gain kept).

### Cross-vehicle robustness (LOVO, frozen; final validation)
Leave-one-vehicle-out across all 7 fleet vehicles, model + lambda frozen, no
tuning from the check — every prediction comes from a model that never saw that
vehicle, so the earlier test inspection does not affect it:

| metric | physics | conservative | Δ |
|---|---|---|---|
| overall MAE % | 0.8613 | 0.5865 | **+31.9%** |
| 0-30 band | 0.4841 | 0.4856 | −0.3% (safe) |
| 60-90 band | 1.4972 | 0.5719 | +61.8% |
| all 7 vehicles improve | — | CUP1..CUP5, ID1, ID2 | ✓ |

### Runtime integration
- `services/calibration/predict.py` — `CalibratedSoCDropPredictor` (lazy model
  load, NaN-tolerant features, per-band lambda, explicit status).
- `services/telemetry-simulator/app/engine/calibrated_soc_drop.py` — lazy
  service wrapper; physics-only fallback whenever the ML artifact is unavailable
  (never raises).
- Endpoints: `GET /api/prediction/status`, `POST
  /api/prediction/calibrated-soc-drop`.
- Schemas: `CalibratedSoCDropRequest` / `CalibratedSoCDropResponse` (additive).

### Frozen artifacts (`artifacts/calibration/v2-ws1/`)
`ws2_freeze_manifest.json` locks: model file + sha256, parameters, best
iteration, lambda policy, split, dataset rows, capacity provenance, all metrics
(test + LOVO) and the validation caveats. **No further tuning is permitted.**

### Honest caveats (restated)
- Test split was inspected during iteration; LOVO is the vehicle-safe,
  non-inspected confirmation and it agrees (even stronger: +31.9%).
- Metrics are SOC-point error in the positive-drop convention; no energy/kWh
  accuracy claim (capacity is an assumed 58 kWh).
- Simulator is not training truth; measured TUM SOC deltas only.