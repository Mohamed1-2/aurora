"""Tests for the AURORA v2 dataset pipeline (real-telemetry ingestors, scenario
builder, split, validation). These use small synthetic-drive fixtures so they run
quickly and offline; the real TUM/EPFL data is validated end-to-end in the build.

Run from repo root:
    .venv/bin/python -m pytest tests/test_dataset_pipeline_v2.py -q
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.evaluation.data.v2.base import ChargingSession, DrivingWindow, Provenance  # noqa: E402
from services.evaluation.data.v2.scenarios import (  # noqa: E402
    build_charging_examples,
    build_driving_examples,
    example_row,
)
from services.evaluation.data.v2.pipeline import (  # noqa: E402
    apply_split_by_unit,
    dedup,
    validate,
    validate_row,
)


def _driving_window(**overrides) -> DrivingWindow:
    base_signals = {
        "soc_pct": 55.0,
        "speed_kph": 60.0,
        "battery_temp_c": 20.0,
        "coolant_temp_c": 20.0,
        "outside_temp_c": 10.0,
        "pack_voltage_v": 400.0,
    }
    base_signals.update(overrides.get("signals", {}))
    prov = Provenance(
        source_dataset="TUM_UDS",
        source_record="data/uds_data/CUP1.parquet",
        state="driving",
        vehicle_id="CUP1",
        signals_used=sorted(base_signals),
        coherence="timestamp_aligned",
        state_reconstruction="temporal_binned_window",
        window_start_utc="2023-01-01T00:00:00+00:00",
        window_duration_s=10.0,
        episode_id=overrides.get("episode_id", "CUP1-2023-01-01"),
    )
    return DrivingWindow(
        vehicle_id="CUP1",
        window_start_utc="2023-01-01T00:00:00+00:00",
        window_duration_s=10.0,
        episode_id="CUP1-2023-01-01",
        signals=base_signals,
        provenance=prov,
    )


def _charging_session(session_id="1", **kw) -> ChargingSession:
    s = {
        "soc_arrival_pct": 30.0,
        "soc_departure_pct": 80.0,
        "stay_min": 40.0,
        "energy_charged_kwh": 30.0,
        "max_power_kw": 80.0,
        "max_requested_power_kw": 120.0,
        "controlled_session": 1.0,
        "total_capacity_kwh": 78.9,
        "bulk_capacity_kwh": 63.0,
    }
    s.update(kw.get("summary", {}))
    prov = Provenance(
        source_dataset="EPFL_L3_CHARGING",
        source_record="Measurement_data.xlsx / Session_data.xlsx",
        state="charging",
        vehicle_id=None,
        signals_used=["soc_pct", "charging_power_kw"],
        coherence="timestamp_aligned",
        state_reconstruction="measured_session",
        session_id=session_id,
    )
    return ChargingSession(
        session_id=session_id,
        measurements=[{"soc_pct": 30.0}, {"soc_pct": 35.0}],
        summary=s,
        provenance=prov,
    )


class TestDrivingScenarioBuilder:
    def test_builds_multiple_examples(self):
        windows = [_driving_window(episode_id=f"CUP1-2023-01-0{i}") for i in range(5)]
        rows = build_driving_examples(windows, random.Random(1), cap_per_window=2)
        assert len(rows) >= 5
        for r in rows:
            assert r["instruction"]
            assert r["input"]
            assert r["output"]
            assert r["provenance"]["source_dataset"] == "TUM_UDS"
            assert r["provenance"]["coherence"] == "timestamp_aligned"
            assert r["provenance"]["data_type"] == "real_telemetry"
            assert r["provenance"]["qa_type"] == "synthetic_generated"

    def test_grounded_answer_matches_context(self):
        w = _driving_window(signals={"soc_pct": 12.0, "speed_kph": 80.0})
        rows = build_driving_examples([w], random.Random(1), cap_per_window=5)
        # At least one soc_level answer must reference ~12% SOC
        soc_rows = [r for r in rows if r["scenario"] == "soc_level"]
        assert soc_rows, "expected a soc_level scenario"
        assert "12" in soc_rows[0]["output"] or "12.0" in soc_rows[0]["output"]

    def test_scenario_name_varies_with_data(self):
        # Directly exercise the soc_level generator for deterministic low-SOC behavior.
        from services.evaluation.data.v2.scenarios import _b_soc_level

        low = _driving_window(signals={"soc_pct": 5.0, "speed_kph": 80.0})
        ex = _b_soc_level(low, random.Random(2))
        assert "low" in ex["output"].lower() or "charg" in ex["output"].lower()
        assert ex["anchor_signals"] == ["soc_pct", "speed_kph"]


class TestChargingScenarioBuilder:
    def test_builds_charging_examples(self):
        cs = _charging_session()
        rows = build_charging_examples([cs], random.Random(3), cap_per_session=4)
        assert len(rows) >= 2
        for r in rows:
            assert r["provenance"]["source_dataset"] == "EPFL_L3_CHARGING"
            assert r["provenance"]["coherence"] == "timestamp_aligned"
            assert r["provenance"]["session_id"] == "1"

    def test_energy_answer_reflects_summary(self):
        cs = _charging_session(summary={"energy_charged_kwh": 30.0})
        rows = build_charging_examples([cs], random.Random(4), cap_per_session=6)
        energy = [r for r in rows if r["scenario"] == "energy_received"]
        assert energy, "expected energy_received scenario"
        assert "30" in energy[0]["output"]

    def test_controlled_power_explanation(self):
        cs = _charging_session(summary={"controlled_session": 1.0,
                                        "max_power_kw": 50.0,
                                        "max_requested_power_kw": 100.0})
        rows = build_charging_examples([cs], random.Random(5), cap_per_session=6)
        pwr = [r for r in rows if r["scenario"] == "power_lower_than_expected"]
        assert pwr, "expected power_lower_than_expected"
        assert "limited" in pwr[0]["output"].lower() or "cap" in pwr[0]["output"].lower() or "station" in pwr[0]["output"].lower()


class TestSplitByUnit:
    def test_no_unit_crosses_splits(self):
        # 3 episodes, several rows each
        windows = [_driving_window(episode_id=f"CUP1-2023-01-0{i}") for i in range(3)]
        rows = build_driving_examples(windows, random.Random(6), cap_per_window=4)
        rows = apply_split_by_unit(rows, random.Random(7))
        unit_split = {}
        for r in rows:
            unit = r["provenance"]["episode_id"]
            if unit in unit_split:
                assert unit_split[unit] == r["split"], "unit leaked across splits"
            else:
                unit_split[unit] = r["split"]

    def test_units_shuffled_but_splits_assigned(self):
        windows = [_driving_window(episode_id=f"CUP1-2023-01-0{i}") for i in range(20)]
        rows = build_driving_examples(windows, random.Random(8), cap_per_window=2)
        rows = apply_split_by_unit(rows, random.Random(9))
        splits = {r["split"] for r in rows}
        assert splits <= {"train", "val", "test"}
        # should produce multiple splits
        assert len({r["split"] for r in rows}) >= 2


class TestValidation:
    def test_valid_row_passes(self):
        r = example_row(
            instruction="sys",
            input="q",
            output="a",
            context={"soc_pct": 50.0},
            intent="battery",
            safety_tier="read_only",
            provenance={"source_dataset": "TUM_UDS", "source_record": "x",
                        "state": "driving", "vehicle_id": "CUP1",
                        "signals_used": ["soc_pct"], "coherence": "timestamp_aligned",
                        "state_reconstruction": "temporal_binned_window",
                        "data_type": "real_telemetry", "qa_type": "synthetic_generated",
                        "human_verified": False},
            derived={},
            scenario="soc_level",
        )
        assert validate_row(r) == []

    def test_missing_field_fails(self):
        r = example_row(
            instruction="sys", input="q", output="a", context={},
            intent="battery", safety_tier="read_only",
            provenance={"source_dataset": "TUM_UDS", "source_record": "x",
                        "state": "driving", "vehicle_id": None,
                        "signals_used": [], "coherence": "timestamp_aligned",
                        "state_reconstruction": "temporal_binned_window",
                        "data_type": "real_telemetry", "qa_type": "synthetic_generated",
                        "human_verified": False},
            derived={}, scenario="soc_level",
        )
        # drop provenance key to force a missing-field error
        del r["provenance"]["coherence"]
        errs = validate_row(r)
        assert any("provenance missing" in e for e in errs)

    def test_bad_data_type_fails(self):
        r = example_row(
            instruction="sys", input="q", output="a", context={},
            intent="battery", safety_tier="read_only",
            provenance={"source_dataset": "X", "source_record": "x",
                        "state": "driving", "vehicle_id": None,
                        "signals_used": [], "coherence": "timestamp_aligned",
                        "state_reconstruction": "temporal_binned_window",
                        "data_type": "fake", "qa_type": "synthetic_generated",
                        "human_verified": False},
            derived={}, scenario="soc_level",
        )
        assert any("data_type" in e for e in validate_row(r))

    def test_pii_sweep(self):
        r = example_row(
            instruction="sys", input="call me at 555-123-4567", output="ok", context={},
            intent="battery", safety_tier="read_only",
            provenance={"source_dataset": "TUM_UDS", "source_record": "x",
                        "state": "driving", "vehicle_id": "CUP1",
                        "signals_used": [], "coherence": "timestamp_aligned",
                        "state_reconstruction": "temporal_binned_window",
                        "data_type": "real_telemetry", "qa_type": "synthetic_generated",
                        "human_verified": False},
            derived={}, scenario="soc_level",
        )
        assert any("PII" in e for e in validate_row(r))


class TestDedup:
    def test_exact_duplicates_removed(self):
        r = {"input": "q", "output": "a", "context": {"x": 1.0}}
        assert len(dedup([dict(r), dict(r)])) == 1

    def test_distinct_contexts_kept(self):
        r1 = {"input": "q", "output": "a", "context": {"x": 1.0}}
        r2 = {"input": "q", "output": "a", "context": {"x": 2.0}}
        assert len(dedup([dict(r1), dict(r2)])) == 2
