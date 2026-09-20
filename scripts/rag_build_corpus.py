"""Build the AURORA retrieval corpus from the committed manifest + evidence store.

Usage:
    python scripts/rag_build_corpus.py [--evidence DIR] [--out DIR] [--skip-analysis]

The corpus landing under out/corpus/ is gitignored (derived from
copyrighted manuals); only the manifest, scripts and tests are committed.
Prints a short summary. Fails loudly on sha256 or page-count violations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services/rag-service"))
MANIFEST = REPO / "services/rag-service/corpus/manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="/Users/mohamedaldofani/Documents/New project/aurora-source-research/evidence")
    parser.add_argument("--out", default=str(REPO / "artifacts/rag"))
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    from app import rag_corpus as rc  # noqa: E402

    evidence_dir = Path(args.evidence).expanduser()
    out_dir = Path(args.out).expanduser()
    manifest = rc.load_manifest(MANIFEST)

    records = rc.build_corpus(manifest, evidence_dir, out_dir)
    by_tier: dict[str, int] = {}
    by_aud: dict[str, int] = {}
    for r in records:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
        by_aud[r.audience] = by_aud.get(r.audience, 0) + 1

    print(f"corpus: {len(records)} chunks -> {out_dir / 'corpus' / 'chunks.jsonl'}")
    print(f"  by_tier: {dict(sorted(by_tier.items()))}")
    print(f"  by_audience: {dict(sorted(by_aud.items()))}")

    if not args.skip_analysis:
        report = rc.near_dup_report(manifest, evidence_dir, out_dir)
        top = sorted(report["summary"], key=lambda s: s["mean_page_sim"], reverse=True)[:3]
        print("near-dup (parked editions):")
        for s in top:
            print(f"  {s['pair']}: mean sim {s['mean_page_sim']}, dup pages {s['dup_pages_share']}")

    info = json.loads((out_dir / "corpus/corpus-info.json").read_text(encoding="utf-8"))
    print(f"corpus sha256: {info['sha256_of_all']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())