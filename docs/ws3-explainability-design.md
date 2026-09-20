# WS3 — Explainability: why did the SOC/arrival prediction change? (design, for review)

Status: **DESIGN ONLY — no implementation in this document.** Reuses WS1
physics outputs + the frozen WS2 promoted predictor. The LLM never computes
numbers; it only verbalises a deterministic structured contribution artifact.

---

## 1. Goal (restated)

AURORA should answer "why did the SOC/arrival prediction change?" using
**structured physics + ML contribution data**, without the LLM inventing
numerical causes. The numbers come from code, not from prose.

Two explanation shapes the system must support:

1. **Absolute** — why is the current 300s-window prediction what it is?
   (physics drivers + ML feature attribution, each in SOC points).
2. **Delta (from→to)** — why did the prediction change between two windows /
   between a baseline and a what-if? (exact per-driver + per-feature deltas).

---

## 2. Principle: "compact ground truth, styled by the LLM"

- A deterministic `PredictionExplanation` object is computed in code from the
  same outputs an LLM would otherwise guess at:
  * **Physics side** — WS1 `PhysicsBaseline.energy_wh()` per-driver split
    (aero / rolling / inertia / aux, hvac=0) converted to SOC points via the
    flagged 58 kWh capacity.
  * **ML side** — the **frozen** LightGBM `calibration_model.lgb` via
    `booster.predict(X, pred_contrib=True)` (exact SHAP attribution:
    17 features + bias, **sums exactly to `ml_residual_pct`** — verified).
  * **Blend** — the same promoted per-band λ (`λ_band`) and the identity
    `calibrated = physics + λ·residual`.
- The LLM receives this as a structured JSON block plus a hard "quote only,
  never compute" instruction, and returns a *stylised sentence*. Numbers in the
  answer are subset-summed/verbatim values from the block; the block is
  *checkable* by a deterministic evaluator (assert quoted figures ⊆ block).

## 3. Reused existing outputs (no new prediction code)

| source | output | role in WS3 |
|---|---|---|
| WS1 `energy_engine.compute_drivers` / `PhysicsBaseline.energy_wh` | aero/rolling/inertia/aux/hvac split (Wh) | physics driver contribution |
| WS2 `CalibratedSoCDropResponse` | `physics_soc_drop_pct`, `ml_residual_pct`, `calibrated_soc_drop_pct`, `speed_band`, `lambda_applied`, `model_version` | the derived numbers being explained |
| WS2 frozen `calibration_model.lgb` | `predict(X, pred_contrib=True)` | exact ML per-feature attribution |
| WS2 `LAMBDA_POLICY_PCT` | per-band λ | effective ML contribution after shrinkage |
| Capacity contract | 58 kWh (assumption flagged) | Wh→SOC-point conversion |

No retraining, no predictor changes, no new physics. The frozen v2-ws1-r1
model and lambda policy are read-only inputs.

## 4. The explanation contract (additive schemas)

New additive pydantic models in `shared/vehicle_ai/schemas.py` (LoRA files
canonical.py / context_prompt.py / normalizer.py / TelemetryFrame are NOT
touched):

```
ContributionItem:  { key, label, value_soc_pct, delta_soc_pct (for delta shape),
                     value_wh, description, provenance ("physics"|"ml") }

PredictionExplanation:
  from_window  |  to_window            # optional second vector => delta shape
  calibrated_soc_drop_pct  physics_soc_drop_pct  ml_residual_pct  speed_band
  lambda_applied  model_version  capacity_kwh
  physics_contributions: [aero, rolling, inertia, aux, hvac=0]   # SOC pts
  ml_contributions:       [16 features, top |contrib| first]     # SOC pts
  bias_term_soc_pct
  identity_check: { physics_plus_ml_equals_calibrated: bool,
                    contributions_sum_to_residual: bool }
  top_causes: [ {cause_label, contribution_soc_pct, direction} ]  # |sorted|
```

- The `top_causes` list is computed by code (|value| sort), so "the main reason
  was X" is grounded, never LLM-inferred.
- `identity_check` is asserted and surfaced, so a broken decomposition is never
  dressed up by prose.

## 5. Identity guarantees (the test before anything is shown to an LLM)

For a single window:
    sum(physics_contributions) == physics_soc_drop_pct
    sum(ml_contributions, incl. bias) == ml_residual_pct        # SHAP identity
    physics_soc_drop_pct + λ_band * ml_residual_pct == calibrated_soc_drop_pct

For a from→to delta:
    Δ_physics = Σ_driver (phys_driver_to − phys_driver_from)     # exact
    Δ_ml      = λ_to * Σ shap_to − λ_from * Σ shap_from          # exact
    Δ_physics + Δ_ml == calibrated_to − calibrated_from          # exact

Unit tests assert all identities at float tolerance. The evaluation then checks
the LLM only quoted figures that exist in the block (see §7).

## 6. Runtime surface — reuse the existing predictor endpoint

Add ONE endpoint next to the existing route (same service, same predictor):

    POST /api/prediction/explain-soc-drop
      body: {"to": {<18 feature dict>}, "from"?: {<18 feature dict>}}
      -> PredictionExplanation

- Backed by `CalibratedSoCDropPredictor` (existing) + a new thin
  `services/calibration/explain.py` that wraps it and adds the SHAP call via
  the already-loaded frozen Booster (`num_iteration=num_trees()`).
- No input is new at the feature level; `from` is optional (absent ⇒ absolute
  shape).
- Fallback identical to WS2: artifact unavailable ⇒
  `physics_only_fallback` explanation (ML contributions absent, physics only,
  status+reason set; never raises).

## 7. LLM integration (assistant prompt, versioned)

- The assistant telemetry pipeline already calls the simulator for prediction
  context; the explanation block is surfaced into the **versioned assistant
  prompt** (`aurora-prompt-v8`, a reviewed diff from v7) as:

      ## PREDICTION EXPLANATION (numbers are exact)
      <PredictionExplanation as compact JSON>
      Rule: quote ONLY values from this block (to 1 decimal place is allowed).
      Never compute ratios, derivatives or totals yourself. If a number you need
      is absent, say you do not have it.

- Intents: add a lightweight `"explain"` trigger (e.g. "why", "explain",
  "why did", "because") so explain questions pull the explanation block, while
  leaving the existing pipeline untouched otherwise.
- No change to the LoRA input/context files or the trained model; the prompt is
  a servable, versioned artifact — the same governance as any prompt change.
- Safety guard rails (rule 4/6 in SYSTEM_PROMPT) remain; explanation cannot be
  used to override policy blocks.

## 8. Evaluation (honesty that prevents invented causes)

1. **Offline unit identities (§5)** — always run, float-exact.
2. **Stub-gateway eval** (pattern already in `test_compare_v2_e2e.py`): ask
   "why did the SOC prediction change?" against a canned `PredictionExplanation`
   and assert:
   - every `\d+(\.\d+)?` number in the answer appears in the block
     (verbatim or 1-dp rounding);
   - no ratio / multiplication / derivative phrase ("×", "per", "3× faster")
     unless the block itself contains a ratio;
   - refusal when the block is `physics_only_fallback` and the question asks
     for ML cause.
3. **Real-data regression**: two real 300s TUM windows → explanation block, both
   shapes (absolute + delta); identity checks green; report top-causes order is
   stable (deterministic).
4. **Metrics recorded** via existing `ai_metrics`: `explanation_identity_ok`,
   `explanation_quote_violations`, `explanation_shape` (absolute|delta).

## 9. Scope / non-goals (kept honest)

- NOT a causal claim about the vehicle: SHAP is attribution of the *model's*
  output; the prompt says "model attributes", never "this caused your battery
  to drain". Physical drivers (aero/rolling/…) are the only causal language.
- NO retraining, NO new prediction model, NO energy claims beyond SOC points
  (kWh stays capacity-flagged).
- NO changes to LoRA boundary files, TelemetryFrame, or the trained model.
- Not implementing assistants' multi-turn reasoning or RAG relevance for
  explanation; that stays later.

## 10. Proposed WS3 implementation steps (stop here for review)

1. `services/calibration/explain.py` — `build_explanation(to_features,
   from_features=None) -> PredictionExplanation` + SHAP wrapper on frozen
   Booster + identity checks.
2. Additive schemas in `shared/vehicle_ai/schemas.py` (§4); no LoRA changes.
3. Endpoint `POST /api/prediction/explain-soc-drop` on the simulator reusing
   the same lazy predictor/fallback.
4. Assistant: `"explain"` intent + prompt v8 `## PREDICTION EXPLANATION` block
   with quote-only rule.
5. Tests: §5 identities (float), §8 stub-gateway quote-violation eval, real
   two-window regression, fallback-shape test.
6. Version bump + docs + report; no training.

**Awaiting your review of §1–§10 (and especially §2, §4, §7, §8) before any
implementation.**