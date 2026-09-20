"""Pydantic contracts shared across all AURORA services.

These schemas are the wire format between the vehicle, the assistant,
the LLM gateway, the evaluation harness and the telemetry simulators.
Keeping them centralized is what lets the platform evolve one service at a time
without breaking the contracts between the others.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DTC(BaseModel):
    """A vehicle Diagnostic Trouble Code (OBD-II style)."""

    code: str
    description: str
    system: str
    severity: Severity = Severity.WARNING


class WarningMessage(BaseModel):
    kind: str
    message: str
    severity: Severity = Severity.WARNING


class GPSPosition(BaseModel):
    lat: float = 0.0
    lon: float = 0.0
    altitude_m: float = 0.0
    heading_deg: float = 0.0


class TelemetryFrame(BaseModel):
    """A single snapshot of vehicle state, emitted every ~100 ms."""

    vehicle_id: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    seq: int
    driving_mode: Literal["eco", "normal", "sport"] = "normal"
    charging: bool = False
    signals: dict[str, float]
    active_dtcs: list[DTC] = Field(default_factory=list)
    warnings: list[WarningMessage] = Field(default_factory=list)
    gps: GPSPosition = Field(default_factory=GPSPosition)

    @property
    def dte_km(self) -> float:
        return self.signals.get("distance_to_empty_km", 0.0)


class TelemetryWindow(BaseModel):
    """Aggregated statistics over the last N frames — used for grounding."""

    vehicle_id: str
    frames: int
    window_seconds: float
    start_ts: float
    end_ts: float
    mean: dict[str, float]
    min: dict[str, float]
    max: dict[str, float]
    std: dict[str, float]
    trending: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    """Result of statistical drift detection against a trained baseline."""

    generated_at: float
    vehicle_id: str
    evaluated_signals: list[str]
    overall_psi: float
    alerted: bool
    per_signal: dict[str, float] = Field(default_factory=dict)
    per_signal_ks_p: dict[str, float] = Field(default_factory=dict)
    strategy: str = "psi-ks-two-stage"


# ---------------------------------------------------------------------------
# LLM Gateway
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class RAGChunk(BaseModel):
    """A retrieved knowledge-base passage that grounds assistant answers."""

    doc_id: str
    section: str
    title: str
    content: str
    score: float
    source: str

    # Provenance (rag-index-v4; backwards-compatible add-on).
    source_id: str | None = None
    edition: str | None = None
    page_pdf: int | None = None
    page_label: str | None = None
    page_range: str | None = None
    tier: str | None = None
    audience: str | None = None
    rights_status: str | None = None
    vehicle_match: str | None = None


class Grounding(BaseModel):
    """Everything non-parametric the assistant may lean on.

    * ``telemetry``  — live aggregated vehicle state (numeric, current).
    * ``documents``  — retrieved manual / policy chunks with provenance.
    * ``events``     — recent DTCs and driver warnings.
    """

    telemetry: dict[str, Any] = Field(default_factory=dict)
    documents: list[RAGChunk] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)


class GatewayRequest(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[Message]
    grounding: Grounding | None = None
    provider: str | None = None          # pin a provider, e.g. "azure-openai"
    model: str | None = None             # pin a model, e.g. "aurora-titan-e"
    max_tokens: int = 600
    temperature: float = 0.2
    tools: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cache_key: str | None = None


class SafetyVerdict(BaseModel):
    """Content-safety + policy verdict produced on every completion."""

    approved: bool = False
    policy: str = "vehicle-assistant-default"
    blocked_reasons: list[str] = Field(default_factory=list)
    score: float = 0.0


class PolicyLevel(str, Enum):
    READ_ONLY = "read_only"
    SAFE_ACTION = "safe_action"
    RESTRICTED_ACTION = "restricted_action"
    PROHIBITED_ACTION = "prohibited_action"


class PolicyDecision(BaseModel):
    """Result of the safety policy gate that runs before every LLM call."""

    level: PolicyLevel = PolicyLevel.READ_ONLY
    reason: str = ""
    matched_pattern: str = ""
    allowed_to_llm: bool = True
    deterministic_answer: str = ""


class GatewayResponse(BaseModel):
    trace_id: str
    provider: str
    model: str
    region: str | None = None
    content: str
    reasoning: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float
    cost_usd: float = 0.0
    fallback_chain: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    safety: SafetyVerdict = Field(default_factory=SafetyVerdict)
    refused: bool = False


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AssistantRequest(BaseModel):
    question: str
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    vehicle_id: str = "ORION-X-TEST-01"
    ground_telemetry: bool = True
    use_rag: bool = True
    stream: bool = False
    prediction_explanation: PredictionExplanation | None = None
    response_style: Literal["standard", "concise"] = "standard"
    conversation: list[ConversationTurn] = Field(default_factory=list, max_length=6)


class GroundingSummary(BaseModel):
    used_telemetry: bool = False
    used_documents: bool = False
    telemetry_signals: list[str] = Field(default_factory=list)
    document_sources: list[str] = Field(default_factory=list)
    events_used: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    question: str
    answer: str
    intents: list[str] = Field(default_factory=list)
    grounding: GroundingSummary = Field(default_factory=GroundingSummary)
    provider: str = ""
    model: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0
    citations: list[str] = Field(default_factory=list)
    risk_level: Severity = Severity.INFO
    session_id: str = ""
    prompt_version: str = ""
    rag_index_version: str = ""
    refused: bool = False
    safety_blocked: list[str] = Field(default_factory=list)
    policy_level: PolicyLevel = PolicyLevel.READ_ONLY


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    expected_signals: list[str] = Field(default_factory=list)
    signal_tolerances: dict[str, float] = Field(default_factory=dict)
    expected_docs: list[str] = Field(default_factory=list)
    expected_intents: list[str] = Field(default_factory=list)
    out_of_scope: bool = False
    expected_policy: PolicyLevel | None = None
    expect_refusal: bool = False
    notes: str = ""


class EvaluationMetric(BaseModel):
    name: str
    value: float
    threshold: float | None = None


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    run_id: str
    timestamp: float
    strategy: str
    prompt_version: str
    rag_index_version: str
    provider: str
    model: str
    environment: str
    aggregate: dict[str, float]
    cases: list[CaseResult]
    mlflow_run_id: str | None = None
    mlflow_registered_model: str | None = None


# ---------------------------------------------------------------------------
# AURORA v2 — Trip & Energy (WS1)
#
# Additive contracts introduced by WS1. They are intentionally separate from
# ``VehicleDrivingContext`` (the LoRA-training contract) and from the legacy
# ``TelemetryFrame``; nothing here changes those schemas.
#
# Honesty boundary (per WS1 decisions): ``ArrivalEstimate`` is a *physics
# projection*, not a real-world accuracy claim. ML calibration over held-out
# real TUM sequences arrives in WS2, and proactive arrival-risk alerts (which
# fire on ``projected_arrival_soc_pct < threshold``) belong to WS4.
# ---------------------------------------------------------------------------


class TripContext(BaseModel):
    """Destination-aware trip configuration for the scenario simulator."""

    destination_km: float = Field(gt=0, description="Trip remaining distance (km)")
    cabin_set_temp_c: float = Field(description="Driver cabin set point (C)")
    ambient_temp_c: float = Field(description="Outside air temperature (C)")
    mode: Literal["scenario", "tum_replay"] = "scenario"
    speed_profile_kph: list[float] = Field(
        default_factory=lambda: [60.0, 90.0, 110.0],
        description="Traveled speed profile for the synthetic scenario trip",
    )
    origin: str = "simulator"
    started_utc: str | None = None


class EnergyFeatures(BaseModel):
    """Breakdown of instantaneous battery power demand (physics baseline)."""

    motive_w: float = 0.0
    aero_w: float = 0.0
    rolling_w: float = 0.0
    inertia_w: float = 0.0
    hvac_w: float = 0.0
    aux_w: float = 0.0
    total_w: float = 0.0


class ArrivalEstimate(BaseModel):
    """Physics projection of arrival state.

    CAUTION: this is a physics consistency estimate for the controlled
    simulator, NOT evidence of real-EV predictive accuracy (WS2 / WS4 deal
    with that separately).
    """

    distance_km: float
    projected_arrival_soc_pct: float
    energy_wh: float
    wh_per_km: float | None = None
    feasibility_note: str = ""


class WhatIfRequest(BaseModel):
    """What-if perturbation over a base steady-state trip projection."""

    distance_km: float = Field(gt=0)
    battery_wh: float = Field(gt=0)
    soc_pct: float
    speed_kph: float = Field(ge=0)
    cabin_temp_c: float = Field(default=18.0, description="Baseline cabin set point (C)")
    ambient_temp_c: float = Field(default=18.0, description="Outside air temperature (C)")
    delta_speed_kph: float = 0.0
    delta_cabin_temp_c: float = 0.0


class WhatIfResponse(BaseModel):
    """Directional result of a what-if (monotonicity testable, not accuracy)."""

    baseline_arrival_soc_pct: float
    perturbed_arrival_soc_pct: float
    delta_arrival_soc_pct: float
    delta_speed_kph: float
    delta_cabin_temp_c: float


class SimulatorTripRequest(BaseModel):
    """POST /api/trip request mapping to vehicle_model.configure_trip()."""

    destination_km: float = Field(gt=0)
    cabin_set_temp_c: float = 22.0
    ambient_temp_c: float = 15.0
    speed_profile_kph: list[float] = Field(
        default_factory=lambda: [60.0, 90.0, 110.0],
        description="Speed profile traveled by the scenario simulator (kph)",
    )
    segment_s: float = Field(default=120.0, ge=1.0)


class TUMReplayRequest(BaseModel):
    """Select a contiguous genuine-measured TUM episode slice for replay."""

    vehicle_id: str = Field(description="Raw parquet vehicle id, e.g. 'ID1'")
    episode_id: str | None = Field(
        default=None,
        description="Provenance episode id, e.g. 'ID1-2022-09-23'. If omitted the longest available episode is used.",
    )
    start_index: int = Field(default=0, ge=0, description="First window index within the episode")
    duration_windows: int | None = Field(default=None, ge=1, description="Number of aligned windows to replay")


class TUMReplaySlice(BaseModel):
    """One replayed genuine window — measured fields only, no fabrication."""

    vehicle_id: str
    window_start_utc: str
    episode_id: str
    signals: dict[str, float]
    canonical: dict[str, float | None]
    provenance: dict[str, Any]
    index: int


# ---------------------------------------------------------------------------
# AURORA v2 — WS2 Calibration prediction contract
#
# Active prediction path (promoted, frozen v2-ws1-r1):
#     calibrated_soc_drop_pct = physics_soc_drop_pct + lam_band * ml_residual_pct
# ``soc_drop_pct`` is positive while discharging. lam_band is the per-speed-band
# shrinkage tuned on validation only. If the ML artifact is unavailable the
# service returns a physics-only prediction with status "physics_only_fallback"
# (never raises).
# ---------------------------------------------------------------------------


class CalibratedSoCDropRequest(BaseModel):
    """Feature vector + physics estimate for a 300s window.

    Missing/None signals are allowed (treated as NaN by the runtime predictor);
    ``physics_soc_drop_pct`` must describe the same window that produced the
    speed/temperature features (positive-drop convention).
    """

    speed_kph_mean: float = 0.0
    speed_kph_std: float = 0.0
    speed_kph_max: float = 0.0
    speed_kph_p90: float = 0.0
    distance_km: float = 0.0
    accel_mean_mps2: float = 0.0
    decel_fraction: float = 0.0
    regen_decel_energy_kwh: float = 0.0
    hv_aux_power_w_mean: float | None = None
    outside_temp_c_mean: float | None = None
    coolant_temp_c_mean: float | None = None
    battery_temp_inlet_c_mean: float | None = None
    battery_temp_outlet_c_mean: float | None = None
    motor_stator_temp_c_mean: float | None = None
    motor_rotor_temp_c_mean: float | None = None
    pack_voltage_v_mean: float | None = None
    physics_soc_drop_pct: float = 0.0


class CalibratedSoCDropResponse(BaseModel):
    """Calibrated SOC-drop prediction with full provenance metadata."""

    physics_soc_drop_pct: float
    ml_residual_pct: float
    calibrated_soc_drop_pct: float
    speed_band: str
    lambda_applied: float
    model_version: str | None
    status: Literal["active", "physics_only_fallback"] = "active"
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# AURORA v2 — WS3 Explainability contract
#
# Deterministic explanation of WHY a SOC-drop prediction is what it is (or why
# it changed between two windows). The numbers and rankings are ALL computed in
# code (physics drivers + SHAP-per-feature on the frozen model); the LLM is only
# a wording layer and must quote these values, never compute its own.
#
# Honesty boundary (per WS3 decisions):
#   * the LightGBM SHAP **bias/base value** is kept separate from user-facing
#     feature causes — it is a model constant, never presented as a physical
#     cause;
#   * contribution identity checks use strict numeric tolerances (not literal
#     float equality);
#   * "ml" provenance is model *attribution*, never a causal claim about the
#     vehicle; only "physics" drivers carry causal language.
# ---------------------------------------------------------------------------


class ContributionItem(BaseModel):
    """One ranked cause of a prediction (physics driver or ML feature).

    ``contribution_soc_pct`` is signed: positive means it increased the
    predicted SOC drop, negative reduced it. ``delta_soc_pct`` carries the
    change for the from→to shape (None for the absolute shape).
    Value is deterministic code output — the LLM may only quote it.
    """

    key: str
    label: str
    contribution_soc_pct: float
    delta_soc_pct: float | None = None
    value_wh: float | None = None
    provenance: Literal["physics", "ml"]
    direction: Literal["increases_drop", "decreases_drop", "neutral"] = "neutral"
    description: str = ""


class PredictionIdentityChecks(BaseModel):
    """Contribution identity checks with strict numeric tolerances."""

    physics_sum_matches: bool = False
    ml_sum_matches: bool = False
    blend_matches: bool = False
    delta_matches: bool | None = None
    tolerance_soc_pct: float = 1e-6


class PredictionExplanation(BaseModel):
    """Explanation of the calibrated SOC-drop prediction (deterministic)."""

    shape: Literal["absolute", "delta"] = "absolute"
    status: Literal["active", "physics_only_fallback"] = "active"
    fallback_reason: str = ""
    model_version: str | None = None
    capacity_kwh: float | None = None
    to_prediction: CalibratedSoCDropResponse
    from_prediction: CalibratedSoCDropResponse | None = None
    physics_contributions: list[ContributionItem] = Field(default_factory=list)
    ml_contributions: list[ContributionItem] = Field(default_factory=list)
    ml_bias_soc_pct: float | None = None  # model base value — separate, NOT a physical cause
    top_causes: list[ContributionItem] = Field(default_factory=list)  # |value| sorted, code-computed
    identity_checks: PredictionIdentityChecks = Field(default_factory=PredictionIdentityChecks)


class ExplainSoCDropRequest(BaseModel):
    """Explain a prediction: ``to`` alone (absolute) or ``from``+``to`` (delta/what-if).

    ``from`` is optional: when present the explanation answers "why did the
    prediction change", otherwise "why is the prediction what it is".
    """

    to: CalibratedSoCDropRequest
    from_: CalibratedSoCDropRequest | None = Field(default=None, alias="from")

    model_config = {"populate_by_name": True}


class ExplainFromTelemetryRequest(BaseModel):
    """Explain a prediction from *raw telemetry frames* (assistant -> simulator).

    The assistant holds live frames (10 Hz) it has already grounded on; it
    forwards the recent window (``to_frames``) and, when it has one, the
    immediately preceding window (``from_frames``) for a delta/what-if
    explanation. The simulator bins the frames and reuses the frozen feature
    extraction + physics baseline, so the feature vector is computed exactly
    once, in the calibration stack.
    """

    to_frames: list[dict[str, Any]]
    from_frames: list[dict[str, Any]] | None = None
    hz: float = 10.0
    capacity_kwh: float = 58.0


# ---------------------------------------------------------------------------
# AURORA v2 — WS4 Proactive intelligence contract
#
# Deterministic alert detection. Code decides whether a vehicle event exists
# from live telemetry + the promoted prediction system; the LLM is only a
# wording layer over an already-decided event and must quote its numbers,
# never invent them.
#
# Honesty boundary (per WS4 decisions):
#   * alerts are STRICTLY signal-gated — a detector never fires for a signal
#     that is absent from the live frame (missing TPMS / range / DTC / etc.
#     means no inference and no alert);
#   * the 300 s calibrated SOC-drop model is treated as a *discharge-rate*
#     signal only. It is never converted into an arrival-SOC / "will you make
#     it" claim unless a trip projection supports that calculation defensibly,
#     so at runtime this stays a high-consumption / discharge-risk alert;
#   * ML numbers are only presented when the prediction ``status=="active"``.
#     ``physics_only_fallback`` downgrades the event severity, marks it
#     ``fallback`` and forbids any ML wording;
#   * every event carries the numeric evidence (observed vs threshold) that the
#     wording layer may quote verbatim but is barred from recomputing.
# ---------------------------------------------------------------------------


class ProactiveTrigger(BaseModel):
    """One numeric alert condition for a deterministic alert (code-computed).

    ``signal`` names the variable that was compared against ``threshold`` and
    MUST be dimensionally consistent with ``observed``/``threshold``/``unit``.
    For change-rate alerts this is the rate variable (e.g.
    ``tire_pressure_rr_kpa_slope_per_min`` with unit ``kPa/min``) — never a raw
    pressure signal whose value would be kPa while the observed value is a rate.
    """

    signal: str
    observed: float | None = None
    threshold: float
    direction: Literal["above", "below"] = "below"
    unit: str = ""
    window_seconds: float | None = None
    samples: int | None = None


class ProactiveEvidence(BaseModel):
    """One observed quantity behind an alert: the raw value the trigger derived.

    Kept separate from ``ProactiveTrigger`` so a value variable (e.g. tyre
    pressure in kPa) and the change-rate variable derived from it (kPa/min) are
    never conflated. The wording layer may quote these verbatim; it must not
    recompute or invent them.
    """

    signal: str
    value: float | None = None
    unit: str = ""
    label: str = ""


class ProactiveEvent(BaseModel):
    """A decided alert. Existence is determined entirely in code."""

    event_id: str
    type: str
    severity: Severity = Severity.WARNING
    status: Literal["full", "partial", "fallback"] = "full"
    title: str
    summary: str
    triggers: list[ProactiveTrigger] = Field(default_factory=list)
    evidence: list[ProactiveEvidence] = Field(default_factory=list)
    explanation: PredictionExplanation | None = None
    signals_used: list[str] = Field(default_factory=list)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    occurrence_count: int = 1
    dismissed: bool = False


class ProactiveSnapshot(BaseModel):
    """Deterministic snapshot served by ``GET /v1/proactive`` (no LLM)."""

    generated_at: float
    vehicle_id: str = ""
    active: list[ProactiveEvent] = Field(default_factory=list)
    recent: list[ProactiveEvent] = Field(default_factory=list)
    detectors_eligible: list[str] = Field(default_factory=list)
    detectors_skipped_no_signal: dict[str, list[str]] = Field(default_factory=dict)
    poll_interval_s: float = 5.0


class ProactiveAskRequest(BaseModel):
    """Words the current top alert(s) through the assistant (option b)."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    vehicle_id: str = "ORION-X-TEST-01"
    count: int = 1
    ack_event_ids: list[str] = Field(default_factory=list)