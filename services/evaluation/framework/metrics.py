"""Evaluation metrics.

Three families, deliberately heterogeneous so a single "good answer" cannot
game the gate on vibes alone:

* **Groundedness (numerical)** — did the answer use the right live signals,
  and does its stated numbers match reality within tolerance?
* **Provenance (retrieval)** — did it actually cite the expected KB docs?
* **Fidelity (text)** — lexical/keyword overlap with a human reference, plus
  an optional LLM-as-judge distillation when a production model is available.
"""

from __future__ import annotations

import re
import statistics

# Loose category tokens for the keyword-overlap metric.
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "in", "on", "for", "and",
    "or", "my", "your", "right", "now", "what", "how", "does", "do", "me",
    "i", "it", "this", "that", "at", "about", "with", "from", "today",
}


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def signal_grounding(
    answer: str,
    used_signals: list[str],
    expected: list[str],
    tolerances: dict[str, float],
    truth: dict[str, float],
) -> dict[str, float]:
    """Score how well live numbers were used, correctly."""
    nums = _numbers(answer)
    checked, passed = 0, 0.0
    for sig in expected:
        if sig not in truth:
            continue
        if sig not in used_signals:
            continue
        checked += 1
        tol = tolerances.get(sig, 5.0)
        target = truth[sig]
        if any(abs(n - target) <= tol for n in nums):
            passed += 1.0
    coverage = (passed / checked) if checked else 1.0
    presence = sum(1 for sig in expected if sig in used_signals) / len(expected) if expected else 1.0
    return {
        "signal_accuracy": round(coverage, 3),
        "signal_presence": round(presence, 3),
        "signal_checked": checked,
    }


def doc_grounding(citations: list[str], expected_docs: list[str]) -> float:
    if not expected_docs:
        return 1.0
    cited_txt = " ".join(citations).lower()
    hits = sum(1 for d in expected_docs if d.lower() in cited_txt)
    return round(hits / len(expected_docs), 3)


def reference_overlap(answer: str, reference: str) -> dict[str, float]:
    if not reference:
        return {"text_recall": 1.0, "lexical_overlap": 1.0}
    ans_tokens = set(re.findall(r"[a-z0-9]+", answer.lower())) - _STOP
    ref_tokens = set(re.findall(r"[a-z0-9]+", reference.lower())) - _STOP
    if not ref_tokens:
        return {"text_recall": 1.0, "lexical_overlap": 1.0}
    matched = ans_tokens & ref_tokens
    recall = len(matched) / len(ref_tokens)
    precision = len(matched) / max(len(ans_tokens), 1)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {
        "text_recall": round(recall, 3),
        "lexical_f1": round(f1, 3),
    }


def refusal_check(answer: str, refused: bool) -> float:
    markers = ["don't have", "dont have", "no reliable data", "can't answer", "cannot answer", "i cannot", "out of scope", "unable"]
    hit = refused or any(m in answer.lower() for m in markers)
    return 1.0 if hit else 0.0


def intent_agreement(response_intents: list[str], expected_intents: list[str]) -> float:
    if not expected_intents:
        return 1.0
    got = set(response_intents)
    want = set(expected_intents)
    return round(len(got & want) / len(want), 3)


def llm_judge_scores() -> dict[str, float]:
    """Placeholder hook. When a production LLM is configured, run it here."""
    return {}


def pxx(values: list[float], x: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * x))]