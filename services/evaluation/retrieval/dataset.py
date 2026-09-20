"""Load and run the AURORA retrieval golden dataset against an index."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from retrieval.metrics import EvalCaseResult, evaluate_case

GRADE_LABELS = {"primary": 2, "secondary": 1}


@dataclass
class ExpectedItem:
    id: str
    doc_id: str
    section_like: str | None
    page_label: str | None
    grade: int


@dataclass
class GoldenCase:
    id: str
    query: str
    audience: str
    expected: list[ExpectedItem]
    primary_lower_tier: bool | None
    has_page_oracle: bool
    baseline: list[str] = field(default_factory=list)


def _as_list(value) -> list[dict]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def load_golden(path: Path, fixture: bool = False) -> list[GoldenCase]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for raw in doc.get("cases", []):
        expected: list[ExpectedItem] = []
        for i, e in enumerate(_as_list(raw.get("expected"))):
            grade = e.get("grade", "primary")
            expected.append(
                ExpectedItem(
                    id=f"{raw['id']}#e{i}",
                    doc_id=e["doc_id"],
                    section_like=e.get("section_like"),
                    page_label=str(e["page_label"]) if e.get("page_label") else None,
                    grade=GRADE_LABELS.get(grade, 2),
                )
            )
        cases.append(
            GoldenCase(
                id=raw["id"],
                query=raw["query"],
                audience=raw.get("audience", "owner"),
                expected=expected,
                primary_lower_tier=raw.get("primary_lower_tier"),
                has_page_oracle=any(e.page_label for e in expected),
                baseline=_as_list(raw.get("baseline")),
            )
        )
    return cases


def evaluate_index(index, cases: list[GoldenCase], top_k: int = 4) -> list[EvalCaseResult]:
    results: list[EvalCaseResult] = []
    for case in cases:
        top = index.retrieve(case.query, top_k=top_k, audience=case.audience)
        hits = [
            {
                "doc_id": r.doc_id,
                "section": r.section,
                "page_label": str(r.page_label) if r.page_label else None,
                "tier": r.tier or "",
                "grade": 0,
                "page_ok": False,
            }
            for r in top
        ]
        for h in hits:
            for e in case.expected:
                if not (h["doc_id"] == e.doc_id and (not e.section_like or e.section_like.lower() in h["section"].lower())):
                    continue
                h["grade"] = max(h["grade"], e.grade)
                if e.page_label is not None and h["page_label"] == e.page_label:
                    h["page_ok"] = True
        results.append(evaluate_case(case, hits, k=top_k))
    return results


def case_summary(result: EvalCaseResult, top_k: int = 4) -> str:
    mark = "PASS" if result.recall >= 1.0 and result.precision_1 and result.dcg >= 0.8 else "fail"
    extra = []
    if result.tier_correct is not None:
        extra.append(f"tier={'ok' if result.tier_correct else 'LOW'}")
    if result.page_correct is not None:
        extra.append(f"page={'ok' if result.page_correct else 'MISS'}")
    failures = " ".join(result.failures) if result.failures else "-"
    return f"{mark:4s} {result.case_id:28s} ndcg={result.dcg:.2f} rec={result.recall:.2f} p1={result.precision_1} mrr={result.mrr_score:.2f} {' '.join(extra):22s} miss: {failures}"