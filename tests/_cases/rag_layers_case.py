"""Behavioural checks for the layered RAG index (corpus + knowledge layer).

Run from the repo root; CI-safe because it uses the committed synthetic
fixture corpus and the committed knowledge-base markdown (no copyrighted
evidence store required). Exercises: layered determinism, provenance of
corpus chunks vs knowledge chunks, audience gating, stopword handling,
knowledge fallback, and deterministic fixture-eval scoring.
"""

import sys

sys.path.insert(0, "services/rag-service")
sys.path.insert(0, "services/evaluation")
sys.path.insert(0, "shared")

from pathlib import Path

from app.index import RagIndex, expand_query, load_kb
from retrieval.dataset import evaluate_index, load_golden

REPO = Path(".").resolve()
KB = REPO / "services/rag-service/knowledge_base"
FIXTURE_CORPUS = REPO / "services/evaluation/retrieval/fixture/corpus.jsonl"
FIXTURE_CASES = REPO / "services/evaluation/retrieval/fixture_cases.yml"


def _layer_sig(index: RagIndex) -> tuple:
    return (
        index.stats.n_chunks,
        index.stats.hash,
        tuple(sorted(index.stats.n_by_tier.items())),
        tuple(sorted(index.stats.n_by_audience.items())),
    )


def _fy(claim: bool) -> tuple[str, bool]:
    return "ok" if claim else "bad"


def main() -> None:
    idx = RagIndex.layered(str(FIXTURE_CORPUS), str(KB))
    rebuilt = RagIndex.layered(str(FIXTURE_CORPUS), str(KB))

    corpus_chunk = next(c for c in idx.chunks if c.source_id)
    knowledge_chunk = next(c for c in idx.chunks if c.doc_id.endswith(".md"))
    knowledge_rag = idx.retrieve("diagnostic trouble code p0420")[0]

    owner_res = idx.retrieve("what must I disconnect first during a rescue", audience="owner", top_k=8)
    responder_res = idx.retrieve("what must I disconnect first during a rescue", audience="responder", top_k=8)
    owner_has_rescue = any(r.doc_id == "fp-rescue-sheet" for r in owner_res)
    responder_has_rescue = any(r.doc_id == "fp-rescue-sheet" for r in responder_res)

    cases = load_golden(FIXTURE_CASES)
    results = evaluate_index(idx, cases, top_k=4)
    results2 = evaluate_index(rebuilt, cases, top_k=4)
    sig1 = tuple((r.case_id, r.dcg, r.recall, r.mrr_score) for r in results)
    sig2 = tuple((r.case_id, r.dcg, r.recall, r.mrr_score) for r in results2)
    all_recall_ok = all(r.recall >= 1.0 for r in results)

    checks = [
        ("layered-deterministic", idx.stats.hash == rebuilt.stats.hash),
        ("layered-tiers-include-knowledge", "knowledge" in (idx.stats.n_by_tier or {})),
        ("owner-gate-excludes-rescuer", not owner_has_rescue),
        ("responder-gate-includes-rescuer", responder_has_rescue),
        ("corpus-chunk-provenance", bool(corpus_chunk.source_id and corpus_chunk.page_pdf and corpus_chunk.tier)),
        ("knowledge-no-source-or-pages", not knowledge_chunk.source_id
         and knowledge_chunk.page_pdf is None and knowledge_chunk.page_label is None),
        ("knowledge-not-manufacturer", knowledge_chunk.tier == "knowledge"
         and "not manufacturer documentation" in knowledge_chunk.rights_status
         and knowledge_chunk.vehicle_match == "unproven"),
        ("retrieval-emits-knowledge-tier", knowledge_rag.tier == "knowledge"
         and not knowledge_rag.source_id),
        ("source-contract-single-hash", all(r.source.count("#") == 1 for r in idx.retrieve("range", top_k=6))),
        ("stopwords-filtered", {"is", "my", "now"} - set(expand_query("What is my estimated range now?")) == {"is", "my", "now"}),
        ("knowledge-fallback", idx.retrieve("diagnostic trouble code p0420")[0].doc_id == "05-diagnostics-and-dtc.md"),
        ("fixture-eval-deterministic", sig1 == sig2),
        ("fixture-eval-recall-ok", all_recall_ok),
    ]
    fails = [name for name, ok in checks if not ok]
    print(f"RAG_LAYER_CHECKS passed={len(checks) - len(fails)} failed={len(fails)}")
    if fails:
        print("FAILED:", ",".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()