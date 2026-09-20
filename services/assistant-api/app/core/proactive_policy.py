"""WS4 proactive policy: detector catalog, thresholds, cooldowns, suppression.

Everything here is plain, reviewable code constants (no heuristics hidden in
prompts) so the alert behaviour is a diff-able, testable artifact — the same
governance the RAG gate and the frozen prediction model follow.

Strict signal-gating contract (per WS4 decisions):
  * every detector declares the signals it requires from the live 10 Hz frame;
  * if ANY required signal is absent at a poll, the detector is skipped and
    produces NO inference and NO alert;
  * a detector only fires once its evidence persists for ``confirm_polls``
    consecutive polls, then honours ``cooldown_s`` and monotonic severity.
"""

from __future__ import annotations

from vehicle_ai.schemas import Severity

TIRE_WHEELS = [
    "tire_pressure_fl_kpa",
    "tire_pressure_fr_kpa",
    "tire_pressure_rl_kpa",
    "tire_pressure_rr_kpa",
]

# The binning stack (``frames_to_bins``) only reads these signals to build the
# 300 s feature vector; the discharge-risk detector gates on exactly this set
# so it can never fire when the prediction window is under-built.
PREDICTION_INPUT_SIGNALS = [
    "vehicle_speed_kph",
    "battery_soc_pct",
    "outside_temp_c",
    "coolant_temp_c",
    "battery_voltage_v",
    "hvac_power_kw",
]

# Critical DTC severities drawn from the vehicle DTC catalog (restraints,
# braking, high voltage) — used for escalation wording only.
CRITICAL_DTC_SYSTEMS = ("restraints", "braking", "high voltage")

DETECTORS: dict[str, dict] = {
    "low_range": {
        # Strict gate: range signal must be present. Suppressed while charging
        # (DTE typically rises on charge, so a low figure is not an alert).
        "required_signals": ["distance_to_empty_km"],
        "unit": "km",
        "threshold": 40.0,               # DTE below this -> alert
        "direction": "below",
        "suppress_while_charging": True,
        "confirm_polls": 2,
        "cooldown_s": 600.0,
        "resolve_after_polls": 3,
        "severity": Severity.WARNING,
        "title": "Low remaining range",
    },
    "tire_pressure_leak": {
        # Per-wheel leak = sustained negative pressure slope. Slope is computed
        # by the engine over a rolling window; this detector only ever reasons
        # about wheels whose own pressure signal is present in the frame.
        "required_signals": TIRE_WHEELS,
        "window_s": 30.0,
        "leak_slope_kpa_per_min": -12.0,  # leak rate (kPa/min) must be this low
        "min_drop_kpa": 3.0,              # ...AND total drop this large (noise guard)
        "confirm_polls": 2,
        "cooldown_s": 900.0,
        "resolve_after_polls": 3,
        "severity": Severity.WARNING,
        "title": "Tyre pressure drop",
    },
    "battery_overtemperature": {
        # Mean battery temperature over the rolling window above ceiling.
        "required_signals": ["battery_temp_c"],
        "window_s": 15.0,
        "max_temp_c": 46.0,
        "escalate_temp_c": 60.0,          # only a genuinely thermal pack is critical
        "confirm_polls": 2,
        "cooldown_s": 900.0,
        "resolve_after_polls": 3,
        "severity": Severity.WARNING,
        "title": "Battery temperature elevated",
    },
    "charging_stall": {
        # Charging flag present & True, but charge_rate ~ 0 over the window.
        "required_signals": ["charge_rate_kw"],
        "window_s": 20.0,
        "max_rate_kw": 0.1,
        "confirm_polls": 2,
        "cooldown_s": 1200.0,
        "resolve_after_polls": 3,
        "severity": Severity.WARNING,
        "title": "Charging stalled",
    },
    "dtc_escalation": {
        # Newly active DTC code (or severity escalation of a known one).
        "required_signals": [],           # gated on the active_dtcs field itself
        "confirm_polls": 1,
        "cooldown_s": 600.0,
        "resolve_after_polls": 3,
        "severity": Severity.WARNING,
        "title": "Diagnostic trouble code",
    },
    "prediction_discharge_risk": {
        # High-consumption / discharge-risk from the promoted 300 s calibrated
        # SOC-drop model. NEVER an arrival-SOC claim: the trip projection is
        # scenario-config at runtime, so we stop at discharge rate.
        "required_signals": PREDICTION_INPUT_SIGNALS,
        "min_frames": 3000,               # full nominal 300 s window @ 10 Hz
        "window_s": 300.0,
        "min_soc_drop_pct": 4.0,          # calibrated drop per 300 s -> alert
        "confirm_polls": 2,
        "cooldown_s": 600.0,
        "resolve_after_polls": 3,
        "severity": Severity.INFO,
        "fallback_severity": Severity.INFO,
        "title": "High energy consumption",
    },
}


def detector_config(detector: str) -> dict | None:
    return DETECTORS.get(detector)