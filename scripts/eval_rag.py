"""Run the AURORA retrieval golden evaluation and enforce the deploy gate.

Usage:
    python scripts/eval_rag.py [--corpus PATH] [--top-k K] [--gate/-no-gate]

Builds (or reuses) the derived corpus under artifacts/rag, runs the judged
golden dataset from services/evaluation/retrieval/golden.yml, prints a per-
case table plus aggregate metrics, and writes artifacts/rag/eval/report.json.
Exits non-zero when a gate threshold is missed.

Gate (embeddings stay deferred while the index clears it):
    ndcg@4 >= 0.80, recall@4 >= 0.75, mrr >= 0.85,
    precision@1 >= 0.70, tier-order >= 0.90, page-anchor >= 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))              # repo-root helpers are not needed, but harmless
sys.path.insert(0, str(REPO / "services/rag-service"))
sys.path.insert(0, str(REPO / "services/evaluation"))
sys.path.insert(0, str(REPO / "shared"))

from app import rag_corpus as rc  # noqa: E402
from app.index import RagIndex  # noqa: E402
from retrieval.dataset import evaluate_index, case_summary, load_golden  # noqa: E402
from retrieval.metrics import aggregate  # noqa: E402

MANIFEST = REPO / "services/rag-service/corpus/manifest.json"
GOLDEN = REPO / "services/evaluation/retrieval/golden.yml"
EVIDENCE = Path("/Users/mohamedaldofani/Documents/New project/aurora-source-research/evidence")

GATE = {
    "ndcg@4": 0.80,
    "recall@4": 0.75,
    "mrr": 0.85,
    "precision@1": 0.70,
    "tier_order": 0.90,
    "page_anchor": 0.80,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(REPO / "artifacts/rag/corpus/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--build", action="store_true", help="force corpus rebuild")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file() or args.build:
        manifest = rc.load_manifest(MANIFEST)
        out_dir = corpus_path.parents[1]
        rc.build_corpus(manifest, Path(args.evidence), out_dir)
        rc.near_dup_report(manifest, Path(args.evidence), out_dir)

    index = RagIndex.from_corpus(str(corpus_path))
    print(f"index: {index.stats.n_chunks} chunks, hash {index.stats.hash}, "
          f"corpus {index.stats.corpus_sha256[:12]}")

    cases = load_golden(GOLDEN)
    results = evaluate_index(index, cases, top_k=args.top_k)
    agg = aggregate(results, k=args.top_k)

    print(f"\n{'case':<30} {'ndcg':>5} {'rec':>5} {'p1':>3} {'mrr':>5}  extra")
    for r in results:
        print(case_summary(r, top_k=args.top_k))

    print("\naggregate:")
    for metric, value in agg.items():
        print(f"  {metric:12s} {value}")
    report = {
        "index_version": f"rag-index-v4-{index.stats.hash}",
        "corpus_sha256": index.stats.corpus_sha256,
        "n_chunks": index.stats.n_chunks,
        "top_k": args.top_k,
        "cases": [
            {
                "id": r.case_id,
                "n_expected": r.n_expected,
                "n_hit": r.n_hit,
                "ndcg": round(r.dcg, 3),
                "recall": round(r.recall, 3),
                "mrr": round(r.mrr_score, 3),
                "precision_at_1": r.precision_1,
                "tier_correct": r.tier_correct,
                "page_correct": r.page_correct,
                "missed_oracles": r.failures,
            }
            for r in results
        ],
        "metrics": agg,
        "gate": GATE,
    }
    out_path = REPO / "artifacts/rag/eval/report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}")

    if args.no_gate:
        return 0
    failures = [m for m, v in agg.items() if m in GATE and v < GATE[m]]
    if failures:
        print("GATE FAILED:", ", ".join(failures))
        return 1
    print("GATE PASSED: every metric clears its threshold; embeddings remain deferred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())