"""Render a canonical vehicle context into the exact prompt template a model was
trained on.

Single source of truth for the "Telemetry: ..." grounding block so the training
notebook, the model-eval harness and the serving provider all compose prompts
byte-for-byte identically. A model that is asked a different prompt at serve time
than it was trained on exhibits a *prompt skew* every bit as harmful as schema
skew — keeping this in one module prevents that.

The telemetry line only includes the fields that are present. Missing fields are
simply omitted (not fabricated), so a driving row and a charging row each render
only their own canonical fields, which is exactly what the model expects.
"""

from __future__ import annotations

from .canonical import (
    VehicleChargingSessionContext,
    VehicleDrivingContext,
)

PREAMBLE = (
    "You are AURORA, an in-vehicle assistant grounded in the AURORA. "
    "Answer only from the provided telemetry. If data is missing, refuse honestly."
)

# Canonical field name -> float-safe renderer used in the telemetry line.
#
# CRITICAL: the training prompt renders the RAW canonical field name exactly as
# written (e.g. ``soc_arrival_pct 12.0``), NOT a human label. The model learned
# from that byte-for-byte format, so serving must reproduce it verbatim or we
# introduce prompt skew. Order follows the canonical schema declaration order.
_DRIVING_FIELDS = (
    "soc_pct",
    "speed_kph",
    "outside_temp_c",
    "battery_temp_c",
    "coolant_temp_c",
    "pack_voltage_v",
)

_CHARGING_FIELDS = (
    "soc_arrival_pct",
    "soc_departure_pct",
    "stay_min",
    "energy_charged_kwh",
    "max_power_kw",
    "max_requested_power_kw",
    "controlled_session",
)


def _kv_pairs(fields: dict[str, float | None], keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key in keys:
        v = fields.get(key)
        if v is None:
            continue
        # Boolean flags render as floats to match the training renderer exactly:
        # v2 stored controlled_session as 1.0/0.0 (float), so the model learned
        # "controlled_session 1.0", not "controlled_session True".
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        out.append(f"{key} {round(float(v), 1)}")
    return out


def telemetry_line(context: VehicleDrivingContext | VehicleChargingSessionContext) -> str:
    """Build the ``Telemetry: ...`` grounding line for a canonical context.

    Returns an empty string when no numeric telemetry is present (the assistant
    is then expected to refuse rather than invent).
    """
    if isinstance(context, VehicleDrivingContext):
        items = _kv_pairs(context.model_dump(), _DRIVING_FIELDS)
    else:
        items = _kv_pairs(context.model_dump(), _CHARGING_FIELDS)
    return "Telemetry: " + "; ".join(items) if items else ""


def build_user_prompt(context: VehicleDrivingContext | VehicleChargingSessionContext, question: str) -> str:
    """Compose the full user block: telemetry (if any) + the question."""
    line = telemetry_line(context)
    return f"{line}\n\nQuestion: {question}" if line else question


def build_messages(
    context: VehicleDrivingContext | VehicleChargingSessionContext,
    question: str,
) -> list[dict[str, str]]:
    """Return the chat messages (system + user) a trained model expects."""
    return [
        {"role": "system", "content": PREAMBLE},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]
