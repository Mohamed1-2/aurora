"""Evaluation metrics used by the CI gate."""

import pytest

from compare import _classify_deltas  # noqa: E402
from framework.metrics import (
    doc_grounding,
    intent_agreement,
    reference_overlap,
    refusal_check,
    signal_grounding,
)


def test_signal_grounding_exact_and_tolerance():
    # truth: dte=251, soc=68 -> answers must contain numbers within tolerance
    out = signal_grounding(
        "Range is 251 km at 68% charge.",
        ["distance_to_empty_km", "battery_soc_pct"],
        ["distance_to_empty_km", "battery_soc_pct"],
        {"distance_to_empty_km": 4, "battery_soc_pct": 2},
        {"distance_to_empty_km": 250.9, "battery_soc_pct": 68.0},
    )
    assert out["signal_accuracy"] == 1.0
    assert out["signal_presence"] == 1.0


def test_signal_grounding_misses_number():
    out = signal_grounding(
        "Range is fine.",
        ["distance_to_empty_km"],
        ["distance_to_empty_km"],
        {"distance_to_empty_km": 4},
        {"distance_to_empty_km": 250.9},
    )
    assert out["signal_accuracy"] == 0.0


def test_signal_grounding_no_expectation_defaults_to_pass():
    out = signal_grounding("no numbers needed", [], [], {}, {})
    assert out["signal_accuracy"] == 1.0
    assert out["signal_presence"] == 1.0


def test_doc_grounding():
    assert doc_grounding(["05-diagnostics-and-dtc.md#Reading"], ["05-diagnostics-and-dtc.md"]) == pytest.approx(1.0)
    assert doc_grounding(["03-range-optimization.md#x"], ["05-diagnostics-and-dtc.md"]) == pytest.approx(0.0)


def test_reference_overlap():
    lo = reference_overlap("the battery is at sixty eight percent", "battery is at 68% SOC")
    hi = reference_overlap("Estimated remaining range is 251 km at 68% charge.", "Estimated remaining range is about 251 km at 68% charge.")
    assert hi["text_recall"] > lo["text_recall"]
    assert 0.0 <= hi["lexical_f1"] <= 1.0


def test_refusal_check():
    assert refusal_check("I don't have reliable data to answer that.", False) == 1.0
    assert refusal_check("Sure, here is the answer.", False) == 0.0
    assert refusal_check("anything", True) == 1.0


def test_intent_agreement():
    assert intent_agreement(["energy", "range"], ["energy"]) == 1.0
    assert intent_agreement(["general"], ["energy"]) == 0.0
    assert intent_agreement([], []) == 1.0


def test_classify_deltas_lower_latency_is_improvement():
    improved, regressed = _classify_deltas(
        {"pass_rate": 0.25, "latency.mean_ms": -1.7}
    )
    assert "latency.mean_ms" in improved
    assert "pass_rate" in improved
    assert regressed == {}


def test_classify_deltas_upper_is_regression():
    improved, regressed = _classify_deltas({
        "pass_rate": -0.1,
        "mean.policy_compliance": 0.0,
        "latency.mean_ms": 5.0,
    })
    assert "pass_rate" in regressed
    assert "latency.mean_ms" in regressed
    assert "mean.policy_compliance" not in improved
    assert "mean.policy_compliance" not in regressed  # zero delta -> neutral