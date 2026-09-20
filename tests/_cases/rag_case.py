"""Behavioural checks for the RAG index, run from the rag-service process."""

import sys

sys.path.insert(0, "services/rag-service")

from pathlib import Path

from app.index import RagIndex, load_kb

KB = Path("services/rag-service/knowledge_base")


def _weighting_boosts_score(idx):
    def score_of(res, doc):
        return next((c.score for c in res if c.doc_id == doc), 0.0)

    plain = idx.retrieve("charging", top_k=10)
    boosted = RagIndex(load_kb(str(KB))).retrieve("charging", top_k=10, doc_weights={"02-battery-and-charging.md": 10.0})
    plain_score = score_of(plain, "02-battery-and-charging.md")
    boost_score = score_of(boosted, "02-battery-and-charging.md")
    return boost_score > plain_score * 9.0


def main():
    idx = RagIndex(load_kb(str(KB)))
    checks = [
        ("kb-loads", idx.stats.n_chunks >= 9 and idx.stats.n_terms > 500 and len(idx.stats.hash) == 16),
        ("deterministic", RagIndex(load_kb(str(KB))).stats.hash == idx.stats.hash),
        ("tyres-query", idx.retrieve("low tyre pressure warning on my wheels")[0].doc_id == "04-tires-and-wheels.md"),
        ("dtc-query", idx.retrieve("diagnostic trouble code p0420")[0].doc_id == "05-diagnostics-and-dtc.md"),
        ("charging-query", idx.retrieve("battery fast charging dc")[0].doc_id == "02-battery-and-charging.md"),
        ("winter-query", idx.retrieve("winter cold range tips")[0].doc_id in {"07-winter-and-weather.md", "03-range-optimization.md"}),
        ("doc-weights-boost", _weighting_boosts_score(idx)),
        ("chunk-contract", idx.retrieve("range", top_k=1)[0].source.count("#") == 1),
    ]
    fails = [name for name, ok in checks if not ok]
    print(f"RAG_CHECKS passed={len(checks) - len(fails)} failed={len(fails)}")
    if fails:
        print("FAILED:", ",".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()