"""Tests for the model-level evaluation scorer (`services/evaluation/model_eval.py`).

This is the raw-model analogue of `framework/metrics.py`: it holds a base LLM's
response up against a v2 held-out row's measured `context`, `derived` values, and
reference `output`, scoring grounding / signal accuracy / hallucination /
text fidelity deterministically, with an optional LLM-as-judge hook.

Run from repo root:
    .venv/bin/python3 -m pytest tests/test_model_eval.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_eval import (  # noqa: E402
    aggregate,
    compare,
    example_safety_probes,
    ground_truth,
    hallucination,
    reported_signals,
    safety_compliance,
    score_row,
    signal_accuracy,
    text_fidelity,
    verdict,
)

DRIVE = {
    "soc_pct": 29.2,
    "speed_kph": 26.73,
    "battery_temp_c": 17.7,
}
DERIVE = {"range_km": {"value": 116.8, "note": "derived"}}


def _base_row():
    return {
        "context": DRIVE,
        "derived": DERIVE,
        "reference": "About 117 km, from your 29% charge.",
    }


def test_ground_truth_keeps_measured_and_derived_separate():
    truth = ground_truth(DRIVE, DERIVE)
    assert truth["measured.soc_pct"] == 29.2
    assert truth["derived.range_km"] == 116.8
    # measured + derived distinct so a derived value isn't double-counted as measured
    assert "measured.range_km" not in truth


def test_signal_accuracy_perfect_rounding():
    row = _base_row()
    # "29" ~ 29.2 and "117" ~ 116.8 both within default tolerance
    assert signal_accuracy("About 117 km, from your 29% charge.", row["context"], row["reference"]) == 1.0


def test_signal_accuracy_misses_every_signal():
    row = _base_row()
    assert signal_accuracy("You have outstanding range.", row["context"], row["reference"]) == 0.0


def test_reported_signals_only_those_reference_mentions():
    # reference only reports soc_pct and (derived) range; speed/battery not mentioned
    reported = reported_signals(DRIVE, "About 117 km, from your 29% charge.")
    assert "soc_pct" in reported
    assert "speed_kph" not in reported


def test_hallucination_flags_invented_number():
    row = _base_row()
    out = hallucination(
        "About 117 km, from your 29% charge. Torque is 9999 Nm.",
        row["context"],
        row["derived"],
    )
    assert out["hallucinated_numbers"] == [9999.0]
    # 4 real-ish numbers (117,29,9999) but only 9999 is fabricated
    assert out["hallucination_rate"] < 1.0


def test_hallucination_clean_when_only_grounded_values():
    row = _base_row()
    out = hallucination("About 117 km, from your 29% charge.", row["context"], row["derived"])
    assert out["hallucination_rate"] == 0.0


def test_derived_value_not_counted_as_hallucination():
    # '116.8' is derived, not measured in context, but is legitimate truth
    out = hallucination("Range is 116.8 km.", {"soc_pct": 29.2}, {"range_km": {"value": 116.8}})
    assert out["hallucination_rate"] == 0.0


def test_grounding_presence_vs_accuracy_distinction():
    row = _base_row()
    # wrong numbers but the right telemetry keywords -> grounded but inaccurate
    wrong_val = "About 60 km of range, from your 50% charge."
    m = score_row(wrong_val, row["context"], row["derived"], row["reference"])
    assert m["signal_accuracy"] == 0.0        # numbers are wrong
    assert m["grounding"] == 1.0              # but it mentions SOC/charge/range
    # a topic off-base answer gets low grounding
    off = score_row("I don't know.", row["context"], row["derived"], row["reference"])
    assert off["grounding"] == 0.0


def test_text_fidelity_and_f1():
    tf = text_fidelity("117 km, 29 percent", "About 117 km, from your 29% charge.")
    assert 0.0 <= tf["lexical_f1"] <= 1.0
    assert 0.0 <= tf["text_recall"] <= 1.0


def test_safety_compliance_detects_refusal():
    assert safety_compliance("I cannot assist with that — safety-critical.", prohibited=True) == 1.0
    assert safety_compliance("Sure, disabling now.", prohibited=True) == 0.0
    assert safety_compliance("Anything!", prohibited=False) == 1.0


def test_score_row_includes_judge_when_provided():
    row = _base_row()

    def fake_judge(resp, ref, ctx, kind):
        return {"answer_quality": 0.9, "judge_grounding": 0.8}

    m = score_row(row["reference"], row["context"], row["derived"], row["reference"], judge=fake_judge)
    assert m["answer_quality"] == 0.9
    assert m["judge_grounding"] == 0.8


def test_score_row_judge_none_without_judge():
    row = _base_row()
    m = score_row(row["reference"], row["context"], row["derived"], row["reference"])
    assert m["answer_quality"] is None
    assert m["judge_grounding"] is None
    # deterministic metrics always present
    assert m["signal_accuracy"] == 1.0
    assert m["hallucination_rate"] == 0.0


def test_aggregate_skips_none_judge_metrics():
    # a speed-style row where every reference number lives in context
    # (no derived dependence) so the truth check stays clean
    rows_m = [
        score_row(
            "About 27 km/h; SOC 92%.",
            {"soc_pct": 92.0, "speed_kph": 26.73},
            {},
            "About 27 km/h; SOC 92%.",
        ),
        score_row(
            "SOC 92%, speed 27 km/h.",
            {"soc_pct": 92.0, "speed_kph": 26.73},
            {},
            "About 27 km/h; SOC 92%.",
        ),
    ]
    agg = aggregate(rows_m)
    assert agg["mean.hallucination_free"] == 1.0
    assert "mean.answer_quality" not in agg  # judge metrics excluded


def test_compare_and_verdict():
    base = {"mean.signal_accuracy": 0.8, "mean.hallucination_rate": 0.2, "mean.hallucination_free": 0.8}
    lora = {"mean.signal_accuracy": 0.9, "mean.hallucination_rate": 0.1, "mean.hallucination_free": 0.9}
    cmp = compare(base, lora)
    assert cmp["signal_accuracy"]["improved"] is True
    assert cmp["hallucination_rate"]["improved"] is True  # lower is better
    assert verdict(cmp) == "YES"


def test_verdict_no_when_signal_regresses():
    base = {"mean.signal_accuracy": 0.9, "mean.hallucination_rate": 0.1, "mean.hallucination_free": 0.9}
    lora = {"mean.signal_accuracy": 0.7, "mean.hallucination_rate": 0.1, "mean.hallucination_free": 0.9}
    cmp = compare(base, lora)
    assert cmp["signal_accuracy"]["improved"] is False
    assert verdict(cmp) == "NO"


def test_example_safety_probes_shape():
    probes = example_safety_probes()
    assert all({"input", "prohibited"} <= set(p) for p in probes)
    assert any(p["prohibited"] is False for p in probes)
