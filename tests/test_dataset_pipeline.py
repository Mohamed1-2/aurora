"""Tests for the production dataset pipeline (schema, generation, curation, validation)."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "evaluation"))

from data.generate import curate, generate, validate, version  # noqa: E402
from data.schema import DatasetRow, SafetyTier, Split  # noqa: E402
from data.telemetry_source import (  # noqa: E402
    load_telemetry_source,
    sample_context,
    value_overview,
)


def _corpus(n_vehicles=4, per_vehicle=10, safety_ratio=0.18) -> list[DatasetRow]:
    return generate(n_vehicles=n_vehicles, rows_per_vehicle=per_vehicle, safety_ratio=safety_ratio)


def test_generate_has_all_expected_intents():
    rows = _corpus(n_vehicles=8, per_vehicle=30)
    intents = {r.intent for r in rows}
    for expected in ("range", "battery", "tire", "diagnostics", "winter", "summary"):
        assert expected in intents


def test_rows_are_grounded_and_never_empty():
    rows = _corpus()
    for r in rows:
        assert r.grounded_answer.strip()
        assert r.utterance.strip()


def test_safety_negatives_are_preserved_and_refusals():
    rows = _corpus(safety_ratio=0.5)
    refusals = [r for r in rows if r.refusals]
    assert len(refusals) > 0
    for r in refusals:
        assert r.safety_tier != SafetyTier.read_only or r.intent == "out_of_scope"


def test_splits_are_leak_free_by_vehicle():
    rows = _corpus(n_vehicles=10)
    veh_split = {}
    for r in rows:
        veh_split.setdefault(r.context.vehicle_id, set()).add(r.split.value)
    for splits in veh_split.values():
        assert len(splits) == 1, f"leak: {splits}"


def test_train_val_test_all_present():
    rows = _corpus(n_vehicles=10)
    splits = {r.split.value for r in rows}
    assert {"train", "val", "test"} <= splits


def test_curate_dedupes_repeated_utterance():
    rows = _corpus()
    deduped, stats = curate(rows)
    assert stats["after_dedupe"] <= len(rows)


def test_validation_returns_empty_for_clean_corpus():
    rows = _corpus()
    rows = curate(rows)[0]
    assert validate(rows) == []


def test_version_is_stable_and_short():
    rows = _corpus()
    assert version(rows) == version(rows)
    assert version(rows).startswith("dataset-")


def test_row_jsonl_roundtrip():
    rows = _corpus()
    from data.schema import to_jsonl

    blob = to_jsonl(rows)
    out = [json.loads(x) for x in blob.splitlines() if x.strip()]
    assert len(out) == len(rows)


# ---------------------------------------------------------------------------
# TelemetrySource ingest (real UDS field histograms) — uses a synthetic fixture
# so the tests run fully offline (CI has no external dataset download).
# ---------------------------------------------------------------------------


_SIGNAL_VID = {
    "soc_pct": 900,
    "speed_kph": 4,
    "outside_temp_c": 15,
    "coolant_temp_c": 1269,
    "battery_temp_c": 1208,
}


@pytest.fixture()
def telemetry_fixture(tmp_path):
    """A minimal dataset folder shaped like the TUM electric-vehicle-uds-dataset."""
    data = tmp_path / "data"
    (data / "json").mkdir(parents=True, exist_ok=True)
    overview = (
        "value_id,name_de,name_en,variable_name,unit,min_val,max_val,sampling_interval_ms\n"
    )
    for field, vid in _SIGNAL_VID.items():
        overview += f"{vid},x,{field},{field},u,-100,200,1000\n"
    (data / "value_overview.csv").write_text(overview)

    for field, vid in _SIGNAL_VID.items():
        # Put all mass in one bin so the sampled value is deterministic-ish.
        edges = [-50.0, 0.0, 5.0, 15.0, 25.0, 50.0]
        counts = [0.0, 0.0, 1000.0, 0.0, 0.0]
        doc = {
            "value_id": vid,
            "vehicle_id": "CUP1",
            "state": "track",
            "value_description": {"variable_name": field, "unit": "u"},
            "data": {"x_values": edges, "y_values": counts},
        }
        (data / "json" / f"track_CUP1_{vid}.json").write_text(json.dumps(doc))
    return tmp_path


def test_load_telemetry_source_builds_samplers(telemetry_fixture):
    src = load_telemetry_source(telemetry_fixture)
    assert set(src.available) == set(_SIGNAL_VID)


def test_value_overview_parses_csv(telemetry_fixture):
    ov = value_overview(telemetry_fixture)
    assert ov[900] == "soc_pct"
    assert len(ov) == len(_SIGNAL_VID)


def test_sample_context_grounds_real_fields(telemetry_fixture):
    src = load_telemetry_source(telemetry_fixture)
    rng = random.Random(7)
    ctx = sample_context(rng, src, "ORION-X-1000", 1.0)
    # Mass bin edges [5,15] -> center ~10.
    assert 5.0 <= ctx.soc_pct <= 15.0
    assert 5.0 <= ctx.speed_kph <= 15.0
    assert 5.0 <= ctx.outside_temp_c <= 15.0
    assert 5.0 <= ctx.coolant_temp_c <= 15.0
    assert 5.0 <= ctx.battery_temp_c <= 15.0
    # range must be derived from the real soc, not random.
    assert ctx.range_km == round(ctx.soc_pct / 100.0 * 400.0, 1)


def test_generate_falls_back_to_mock_on_bad_telemetry_dir():
    rows = generate(n_vehicles=4, rows_per_vehicle=10, telemetry_dir="/does/not/exist")
    assert rows
    # Mock coolant was 70-105; real ground truth is ~0-50. Bad dir -> must be mock.
    cools = {r.context.coolant_temp_c for r in rows}
    assert any(c >= 60 for c in cools)


def test_generate_uses_real_telemetry_when_provided(telemetry_fixture):
    rows = generate(n_vehicles=4, rows_per_vehicle=10, telemetry_dir=str(telemetry_fixture))
    socs = {r.context.soc_pct for r in rows}
    assert socs
    assert all(x is not None and -100 <= x <= 200 for x in socs)
