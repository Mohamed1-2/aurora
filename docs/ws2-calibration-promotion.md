# WS2 — Calibration promotion: cross-vehicle robustness validation (frozen)

Status: **PROMOTED** · version `v2-ws1-r1` · frozen (no further tuning).

## What was promoted
Active prediction path:

```
calibrated_soc_drop_pct = physics_soc_drop_pct + lam_band * ml_residual_pct
```

- `physics_soc_drop_pct`: WS1 wheel-drag physics + measured `hv_aux_power`,
  hvac = 0, on the dense 5-core signal set, 300 s windows (58 kWh assumed).
- `ml_residual_pct`: LightGBM (40 trees) trained on real TUM SOC deltas.
- `lam_band`: per-speed-band shrinkage tuned on **validation only** —
  0-30:0.20, 30-60:1.00, 60-90:1.00, 90-120:1.00, 120+:0.55.

## Headline results

### Held-out test split (ID2+CUP3) — single confirmatory look
| model | overall MAE % (SOC pts) | vs physics |
|---|---|---|
| physics only | 0.8605 | reference |
| conservative (promoted) | **0.6520** | **−24.2%** |
| full LightGBM | 0.6574 | −23.6% (rejected: 0-30 band +11.2% MAE) |

### Cross-vehicle robustness (LOVO, final validation)
Leave-one-vehicle-out across all 7 fleet vehicles with the **same frozen model
hyperparameters, 40 rounds, and frozen lambda policy** — nothing was tuned or
optimized from this check. Every prediction came from a model trained only on
the *other* vehicles' episodes, so the earlier test-split inspection does not
touch these numbers.

| aggregated (pooled LOVO, n=3677) | physics | conservative | improvement |
|---|---|---|---|
| overall MAE % (SOC pts) | 0.8613 | **0.5865** | **+31.9%** |
| RMSE % | 2.1098 | 1.2881 | −39% |
| bias % | −0.2776 | 0.0005 | removed |

Per speed band (LOVO pooled MAE):
| band | physics | conservative | Δ |
|---|---|---|---|
| 0-30 | 0.4841 | 0.4856 | −0.3% (safe) |
| 30-60 | 0.6369 | 0.5305 | +16.7% |
| 60-90 | 1.4972 | 0.5719 | +61.8% |
| 90-120 | 0.9521 | 0.6880 | +27.8% |
| 120+ | 0.7755 | 0.7070 | +8.8% |

Per vehicle (LOVO, all improve): CUP1 0.676→0.436, CUP2 1.179→0.744,
CUP3 1.734→1.547, CUP4 1.010→0.599, CUP5 1.159→1.062, ID1 0.794→0.548,
ID2 0.825→0.543.

## Why this passes the promotion gate
1. Test MAE ≥5% relative improvement: **PASS (+24.2%)**; LOVO agreement +31.9%.
2. No vehicle regresses >+20% on either held-out test or LOVO: **PASS**.
3. No major speed band regresses >+10%: **PASS** — the promoted policy is the
   conservative per-band shrink that fixed the exact band (0-30, was +11.2% under
   full LightGBM) which had originally blocked promotion.

## Validation caveats (must stay attached to these numbers)
1. The held-out test split (ID2, CUP3) was inspected during development; the
   lambda policy was fixed from **validation only**, and the 24.2% test figure is
   a single confirmatory look. Do not re-use that split as a tuning target.
2. The authoritative, non-inspected number is the LOVO result (+31.9% overall,
   band safety holds, every vehicle improves) because it evaluates each vehicle
   with a model that never saw it, under the fully frozen policy.
3. Metrics are **SOC-point (%) errors** in the positive-drop convention
   (`drop = soc_start − soc_end`). No energy/kWh accuracy claim is made: the
   58 kWh capacity is a fleet-level assumption (`tum_dataset_readme_fleet_spec`,
   all assumptions flagged). kWh MAE is capacity-sensitive.
4. Simulator data is never training truth; labels are measured TUM SOC deltas.
5. The physics-only model remains the runtime fallback/reference and is not
   deleted; the calibration degrades to it cleanly (`physics_only_fallback`).

## Freeze
`artifacts/calibration/v2-ws1/ws2_freeze_manifest.json` locks model
(sha256, params, best_iteration), lambda policy, split, dataset rows, capacity
provenance, every metric, and these caveats. **No further tuning is permitted.**

## Runtime surface
- `services/calibration/predict.py` → `CalibratedSoCDropPredictor`
- `services/telemetry-simulator/app/engine/calibrated_soc_drop.py` → lazy wrapper
- Endpoints: `GET /api/prediction/status`, `POST
  /api/prediction/calibrated-soc-drop`
- Schemas: `CalibratedSoCDropRequest` / `CalibratedSoCDropResponse`