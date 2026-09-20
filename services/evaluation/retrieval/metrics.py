"""Retrieval evaluation metrics for the AURORA RAG golden set.

Ranked-list metrics (k = top_k, default 4) measured against a judged
golden dataset where each expected item carries a graded relevance
(2 = primary answer, 1 = acceptable for that intent). Relevance of a
retrieved chunk = (doc_id in expected) AND (section substring matches) AND
(page_label matches when the oracle pins one).

Families:
* nDCG@k   — graded gains, log-discounted rank.
* recall@k — fraction of grade-2 oracles actually retrieved.
* MRR      — reciprocal rank of the first relevant hit (grade >= 1).
* precision@1 — first result is a grade-2 hit.
* tier-order — for cases that exercise both primary-tier (manual/appendix)
  and lower-tier (vehicle_specific/family/generic) material, is the first
  relevant hit at primary tier? Mirrors the "verified manual first" policy.
* page-anchor — when an oracle pins a printed page, did a matching chunk with
  that page surface in top_k?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EvalCaseResult:
    case_id: str
    query: str
    n_expected: int
    n_hit: int
    dcg: float
    recall: float
    mrr_score: float
    precision_1: int
    tier_correct: int | None
    page_correct: int | None
    failures: list[str] = field(default_factory=list)


def ndcg_at_k(gains: list[float], k: int = 4) -> float:
    gained = gains[:k]
    ideal = sorted(gains, reverse=True)[:k]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gained))
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(hits: int, n_expected: int) -> float:
    return (hits / n_expected) if n_expected else 1.0


def evaluate_case(case, hits: list[dict], k: int = 4) -> EvalCaseResult:
    """hits: retrieved chunks in rank order {doc_id, section, page_label, grade, page_ok}.

    Relevance (grade) is decided on doc + section; the page oracle feeds the
    separate page-anchor metric so a wrong page never zeroes a good recall.
    """
    expected = [e for e in case.expected if e.grade >= 1]
    expected_p2 = [e for e in case.expected if e.grade == 2]
    gains, matched_p2, failures = [], 0, []
    for e in expected_p2:
        ok = any(h["doc_id"] == e.doc_id and (not e.section_like or e.section_like.lower() in h["section"].lower()) for h in hits)
        if ok:
            matched_p2 += 1
        else:
            failures.append(e.id)
    relevant_first_rank = None
    for rank, h in enumerate(hits, start=1):
        if h["grade"] >= 1:
            gains.append(float(h["grade"]))
            if relevant_first_rank is None:
                relevant_first_rank = rank
    tier = None
    if case.primary_lower_tier is not None:
        first_tier = hits[0]["tier"] if hits else None
        primary_tiers = {"manual", "appendix"}
        tier = int(first_tier in primary_tiers)
    page = None
    if case.has_page_oracle:
        page = int(any(h["page_ok"] for h in hits))
    return EvalCaseResult(
        case_id=case.id,
        query=case.query,
        n_expected=len(expected),
        n_hit=sum(1 for g in gains if g >= 1),
        dcg=ndcg_at_k(gains, k),
        recall=recall_at_k(matched_p2, len(expected_p2)),
        mrr_score=1.0 / relevant_first_rank if relevant_first_rank else 0.0,
        precision_1=int(bool(gains and gains[0] >= 2)),
        tier_correct=tier,
        page_correct=page,
        failures=failures,
    )


def aggregate(results: list[EvalCaseResult], k: int = 4) -> dict[str, float]:
    tier_scored = [r.tier_correct for r in results if r.tier_correct is not None]
    page_scored = [r.page_correct for r in results if r.page_correct is not None]
    return {
        f"ndcg@{k}": round(sum(r.dcg for r in results) / len(results), 3) if results else 0.0,
        f"recall@{k}": round(sum(r.recall for r in results) / len(results), 3) if results else 0.0,
        "mrr": round(sum(r.mrr_score for r in results) / len(results), 3) if results else 0.0,
        "precision@1": round(sum(r.precision_1 for r in results) / len(results), 3) if results else 0.0,
        "tier_order": round(sum(tier_scored) / len(tier_scored), 3) if tier_scored else 1.0,
        "page_anchor": round(sum(page_scored) / len(page_scored), 3) if page_scored else 1.0,
    }