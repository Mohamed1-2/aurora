"""Canonical vehicle-context contract + runtime schema normalizer tests.

These guard the training-serving skew boundary: v2/TUM/EPFL/ORION-X fields must
normalise into the exact canonical names the LoRA was trained on, nothing more
and nothing conflated (e.g. soc_arrival_pct is never a stream soc_pct).
"""

import pytest
from pydantic import ValidationError

from vehicle_ai.canonical import (
    VehicleChargingSessionContext,
    VehicleDrivingContext,
    VehicleGroundTruth,
)
from vehicle_ai.context_prompt import (
    PREAMBLE,
    build_messages,
    build_user_prompt,
    telemetry_line,
)


class TestCanonicalContract:
    def test_driving_optional_fields_default_none(self):
        d = VehicleDrivingContext()
        assert d.soc_pct is None
        assert d.speed_kph is None
        assert d.outside_temp_c is None
        assert d.battery_temp_c is None
        assert d.coolant_temp_c is None
        assert d.pack_voltage_v is None

    def test_charging_optional_fields_default_none(self):
        c = VehicleChargingSessionContext()
        assert c.soc_arrival_pct is None
        assert c.soc_departure_pct is None
        assert c.stay_min is None
        assert c.energy_charged_kwh is None
        assert c.max_power_kw is None
        assert c.max_requested_power_kw is None
        assert c.controlled_session is None

    def test_full_driving_context(self):
        d = VehicleDrivingContext(
            soc_pct=66.4,
            speed_kph=23.09,
            outside_temp_c=-4.5,
            battery_temp_c=6.1,
            coolant_temp_c=12.0,
            pack_voltage_v=400.0,
        )
        assert d.soc_pct == 66.4
        assert d.outside_temp_c == -4.5

    def test_controlled_session_bool_accepted(self):
        c = VehicleChargingSessionContext(controlled_session=True)
        assert c.controlled_session is True

    def test_driving_never_holds_charging_fields(self):
        # A driving context must not implicitly absorb charging-session fields.
        with pytest.raises(ValidationError):
            VehicleDrivingContext(soc_arrival_pct=12.0)  # type: ignore[call-arg]

    def test_ground_truth(self):
        g = VehicleGroundTruth(range_km=116.8)
        assert g.range_km == 116.8
        assert VehicleGroundTruth().range_km is None


class TestNormalizer:
    def test_v2_driving_identity(self):
        from vehicle_ai.normalizer import from_v2_driving

        ctx = from_v2_driving(
            {"soc_pct": 66.4, "speed_kph": 23.09, "battery_temp_c": 6.09, "outside_temp_c": -4.5}
        )
        assert isinstance(ctx, VehicleDrivingContext)
        assert ctx.soc_pct == 66.4
        assert ctx.speed_kph == 23.09
        # Absent signals stay None, never invented.
        assert ctx.coolant_temp_c is None
        assert ctx.pack_voltage_v is None

    def test_v2_charging_identity_and_bool(self):
        from vehicle_ai.normalizer import from_v2_charging

        c = from_v2_charging(
            {"soc_arrival_pct": 12.0, "soc_departure_pct": 91.0, "controlled_session": 1.0}
        )
        assert isinstance(c, VehicleChargingSessionContext)
        assert c.soc_arrival_pct == 12.0
        assert c.soc_departure_pct == 91.0
        assert c.controlled_session is True
        assert c.stay_min is None

    def test_soc_arrival_is_never_stream_soc(self):
        # The critical semantic boundary: arrival SOC is NOT the current SOC.
        from vehicle_ai.normalizer import from_v2_charging

        c = from_v2_charging({"soc_pct": 54.0, "soc_arrival_pct": 12.0})
        assert c.soc_arrival_pct == 12.0
        assert not hasattr(c, "soc_pct")  # charging context holds no current SOC

    def test_orionx_live_maps_legacy_names(self):
        from vehicle_ai.normalizer import from_orionx_live

        d = from_orionx_live(
            {
                "signals": {
                    "battery_soc_pct": 68.5,
                    "vehicle_speed_kph": 41.2,
                    "battery_temp_c": 9.3,
                    "coolant_temp_c": 8.0,
                }
            }
        )
        assert d.soc_pct == 68.5
        assert d.speed_kph == 41.2
        assert d.battery_temp_c == 9.3
        assert not hasattr(d, "vehicle_speed_kph")  # legacy name is not canonical

    def test_orionx_live_flat_payload(self):
        from vehicle_ai.normalizer import from_orionx_live

        d = from_orionx_live({"battery_soc_pct": 20.5})
        assert d.soc_pct == 20.5

    def test_dispatch_by_source(self):
        from vehicle_ai.normalizer import normalizer_for

        n = normalizer_for("TUM_UDS")
        assert n({"soc_pct": 10.0}).soc_pct == 10.0
        c = normalizer_for("EPFL_L3_CHARGING", charging=True)
        assert isinstance(c({"soc_arrival_pct": 5.0}), VehicleChargingSessionContext)

    def test_dispatch_unknown_source_raises(self):
        from vehicle_ai.normalizer import normalizer_for

        with pytest.raises(KeyError):
            normalizer_for("no-such-source")

    def test_nan_guard(self):
        from vehicle_ai.normalizer import from_v2_driving

        d = from_v2_driving({"soc_pct": float("nan")})
        assert d.soc_pct is None
        assert from_v2_driving({"soc_pct": None}).soc_pct is None
        assert from_v2_driving({"soc_pct": "garbage"}).soc_pct is None

    def test_ground_truth_derived(self):
        from vehicle_ai.normalizer import ground_truth_from_v2

        gt = ground_truth_from_v2({"derived": {"range_km": {"value": 116.8, "note": "..."}}})
        assert gt.range_km == 116.8
        assert ground_truth_from_v2({}).range_km is None


class TestContextPrompt:
    def test_preamble_matches_training(self):
        assert PREAMBLE == (
            "You are AURORA, an in-vehicle assistant grounded in the AURORA. "
            "Answer only from the provided telemetry. If data is missing, refuse honestly."
        )

    def test_driving_line_renders_only_present_fields(self):
        from vehicle_ai.normalizer import from_v2_driving

        ctx = from_v2_driving(
            {"soc_pct": 66.4, "speed_kph": 23.09, "battery_temp_c": 6.09, "outside_temp_c": -4.5}
        )
        assert telemetry_line(ctx) == (
            "Telemetry: soc_pct 66.4; speed_kph 23.1; outside_temp_c -4.5; battery_temp_c 6.1"
        )

    def test_charging_line_includes_controlled_session_float(self):
        from vehicle_ai.normalizer import from_v2_charging

        ctx = from_v2_charging({"max_power_kw": 96.009, "max_requested_power_kw": 353.265, "controlled_session": 1.0})
        assert telemetry_line(ctx) == (
            "Telemetry: max_power_kw 96.0; max_requested_power_kw 353.3; controlled_session 1.0"
        )

    def test_user_prompt_with_context(self):
        from vehicle_ai.normalizer import from_v2_driving

        ctx = from_v2_driving({"soc_pct": 66.4})
        assert build_user_prompt(ctx, "Give me the current vehicle summary.") == (
            "Telemetry: soc_pct 66.4\n\nQuestion: Give me the current vehicle summary."
        )

    def test_user_prompt_without_context_is_just_question(self):
        d = VehicleDrivingContext()
        assert build_user_prompt(d, "How is my car?") == "How is my car?"

    def test_messages_shape(self):
        from vehicle_ai.normalizer import from_v2_charging

        ctx = from_v2_charging({"energy_charged_kwh": 40.885})
        msgs = build_messages(ctx, "Summarize my charging session.")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == PREAMBLE
        assert msgs[1]["role"] == "user"
        assert "energy_charged_kwh 40.9" in msgs[1]["content"]