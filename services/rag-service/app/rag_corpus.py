"""Corpus synthesis for the AURORA knowledge base.

Builds a provenanced, deterministic retrieval corpus from the source register
(EVidence atlas) and the committed manifest at services/rag-service/corpus.

Design rules:
* Only sources marked ``ingested: true`` become retrieval chunks.
* Text is re-extracted from the original PDF with pymupdf when available
  (the pre-built .txt fallback carries mid-word split noise like
  "Char ging" / "pag e 174" that degrades BM25); pymupdf output is clean.
* Each PDF's sha256 is verified against the manifest before ingestion.
* Chunks carry page tracking (printed label + PDF page), the running section
  heading, tier, audience and rights status so the index can do
  tier/audience-aware retrieval and the UI can show provenance.
* The whole pipeline is deterministic (no randomness, no timestamps), so a
  rebuild yields byte-identical chunks and a stable index hash.
* Near-duplicate analysis across the parked editions (ingested=false) is
  written to a report; those editions are never used for retrieval.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # Build-time only; services never import pymupdf.
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal envs
    pymupdf = None

_PDF_PAGE_RE = re.compile(r"^\s*PDF PAGE (\d+)\s*$", re.MULTILINE)
_HYPHEN_SPLIT_RE = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])")
_CROSSREF_RE = re.compile(r"[›»]{1,3}\s*(?:page|pag e|Fig\.?|Section|→)*\s*\d+(?:[–-]\d+)?")
_BULLET_RE = re.compile(r"[•●▪]")
_NAV_SECTION_RE = re.compile(
    r"^(Table of Contents|Index|General views of the vehicle|Interior view|"
    r"Overview \(.*\)|About this instruction manual( .*)?)$"
)
_SHORT_TITLE_MAX = 60

TIER_ORDER = ["manual", "appendix", "vehicle_specific", "family", "generic"]
# Authority multiplier applied on top of BM25 (tie-breaker, not dominant).
TIER_MULTIPLIER = {
    "manual": 1.7,
    "appendix": 1.6,
    "vehicle_specific": 1.35,
    "family": 1.15,
    "generic": 1.0,
}


class CorpusError(RuntimeError):
    """Raised when a committed manifest contract is violated at build time."""


@dataclass
class Page:
    index: int  # 1-based PDF page index
    label: str | None  # printed page number, when detectable
    title: str  # running section heading detected for this page
    text: str  # cleaned page text


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_id: str
    title: str
    edition: str
    section: str
    page_pdf: int
    page_label: str | None
    page_range: str
    content: str
    tier: str
    audience: str
    rights_status: str
    vehicle_match: str
    source: str
    sha256: str

    def __post_init__(self) -> None:
        if self.tier not in TIER_ORDER:
            raise CorpusError(f"unknown tier {self.tier!r} in {self.chunk_id}")

    def to_json(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "title": self.title,
            "edition": self.edition,
            "section": self.section,
            "page_pdf": self.page_pdf,
            "page_label": self.page_label,
            "page_range": self.page_range,
            "content": self.content,
            "tier": self.tier,
            "audience": self.audience,
            "rights_status": self.rights_status,
            "vehicle_match": self.vehicle_match,
            "source": self.source,
            "sha256": self.sha256,
        }


def clean_text(text: str) -> str:
    """Normalise extracted PDF text into searchable prose.

    Repairs hyphen line-break splits, strips cross-reference stamps
    (``››› page 174``), turns bullet glyphs into ``- `` markers, collapses
    whitespace, and drops page-marker noise.
    """
    text = text.replace("\u00ad", "")
    text = _HYPHEN_SPLIT_RE.sub("", text)
    text = _CROSSREF_RE.sub("", text)
    text = text.replace("›", " ").replace("»", " ")
    text = _BULLET_RE.sub("- ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _extract_with_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(1-based page index, raw page text)] using pymupdf."""
    doc = pymupdf.open(str(pdf_path))
    try:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    finally:
        doc.close()


def _extract_with_txt(pdf_path: Path) -> list[tuple[int, str]]:
    """Fallback: parse pre-built 'PDF PAGE N' extraction when pymupdf is absent."""
    txt_path = pdf_path.with_suffix(pdf_path.suffix + ".txt")
    if not txt_path.exists():
        raise CorpusError(f"no pymupdf and no pre-built extraction for {pdf_path.name}")
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    parts = _PDF_PAGE_RE.split(raw)
    # parts[0] is the preamble; then pairs (page_no, text).
    pages: list[tuple[int, str]] = []
    for page_no, body in zip(parts[1::2], parts[2::2]):
        pages.append((int(page_no), body))
    return pages


def extract_pages(pdf_path: Path, page_count: int | None = None) -> list[tuple[int, str]]:
    if pymupdf is not None:
        pages = _extract_with_pymupdf(pdf_path)
    else:
        pages = _extract_with_txt(pdf_path)
    if page_count is not None and len(pages) != page_count:
        raise CorpusError(
            f"{pdf_path.name}: extracted {len(pages)} pages, manifest expects {page_count}"
        )
    return pages


def _detect_heading(lines: list[str]) -> tuple[str | None, str | None]:
    """Find the (running title, printed page label) at the top of a page.

    CUPRA/ID manuals print the running section title on a line, followed by a
    bare printed page number on the next line. A heading must start with an
    uppercase letter (not a digit), stay under a length cap, and not carry
    sentence-ending punctuation (prose looks different from headings).
    """
    for i, line in enumerate(lines):
        if not line or line.isdigit():
            continue
        candidate = line.strip()
        if len(candidate) > _SHORT_TITLE_MAX:
            continue
        if not re.match(r"^[A-Z][A-Za-z ,'&()/.:‘’-]+$", candidate):
            continue
        if candidate[-1] in ". ,;:":
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt.isdigit():
            return candidate, nxt
    return None, None


def _index_page_noise(text: str) -> bool:
    """Heuristic: is this an index/TOC page (dotted tab leaders) to skip?"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    dotted = sum(1 for ln in lines if re.search(r"\.{4,}", ln))
    return dotted / len(lines) > 0.2


def pages_from_pdf(
    pdf_path: Path,
    page_count: int | None,
    fallback_title: str | None = None,
) -> list[Page]:
    """Extract, clean and annotate pages with running section + page label.

    Skips cover/TOC/index/figure-only pages and carries the running title
    across pages that do not start with a detectable heading pair. Documents
    without the manual heading layout (technical reports / family guides) fall
    back to ``fallback_title`` for their section label.
    """
    out: list[Page] = []
    current_title: str | None = None
    for idx, raw in extract_pages(pdf_path, page_count):
        text = clean_text(raw)
        if not text or _index_page_noise(text):
            continue
        if len(text) < 80:  # figure-only / stamp pages
            continue
        lines = text.splitlines()
        title, label = _detect_heading(lines)
        if title:
            current_title = title
        if current_title is None and fallback_title is not None:
            current_title = fallback_title
        if current_title is None:
            continue  # un-attributable front matter (cover / title page)
        if _NAV_SECTION_RE.match(current_title):
            continue  # browsing/index hubs that crowd real procedures
        out.append(Page(index=idx, label=label, title=current_title, text=text))
    return out


def _chunk_pages(pages: list[Page], max_chars: int = 2200) -> list[tuple[list[Page], str]]:
    """Group pages into chunks by (section, size), splitting on page marks.

    Returns (pages_in_chunk, section_name). Deterministic and bounded.
    """
    chunks: list[tuple[list[Page], str]] = []
    current: list[Page] = []
    section: str | None = None
    size = 0
    for page in pages:
        if section is not None and page.title != section:
            if current:
                chunks.append((current, section))
                current, section, size = [], None, 0
            section = page.title
        elif section is None:
            section = page.title
        if page.text:
            current.append(page)
            size += len(page.text)
        if size >= max_chars:
            chunks.append((current, section))
            current, section, size = [], None, 0
    if current:
        chunks.append((current, section or ""))
    return chunks


def _page_label(pages: list[Page]) -> str | None:
    for p in pages:
        if p.label is not None and p.label.isdigit() and int(p.label) <= p.index + 250:
            return p.label
    return None


def build_corpus(
    manifest: dict,
    evidence_dir: Path,
    out_dir: Path,
    max_chars: int = 2200,
) -> list[ChunkRecord]:
    """Synthesise the retrieval corpus from the manifest + evidence store.

    Returns chunk records, writes ``corpus/chunks.jsonl`` and
    ``corpus/corpus-info.json`` under ``out_dir``. Idempotent and
    deterministic.
    """
    if not evidence_dir.is_dir():
        raise CorpusError(f"evidence dir missing: {evidence_dir}")
    records: list[ChunkRecord] = []
    for src in manifest["sources"]:
        if not src.get("ingested", False):
            continue
        pdf_path = evidence_dir / src["evidence_file"]
        if not pdf_path.exists():
            raise CorpusError(f"evidence file missing for {src['source_id']}: {pdf_path}")
        actual = sha256_file(pdf_path)
        expected = src.get("sha256")
        if expected is not None and actual != expected:
            raise CorpusError(
                f"sha256 mismatch for {src['source_id']}: got {actual}, manifest says {expected}"
            )
        fallback = src["title"] if src["tier"] in ("family", "generic") else None
        pages = pages_from_pdf(pdf_path, src.get("page_count"), fallback)
        for page_group, section in _chunk_pages(pages, max_chars=max_chars):
            first, last = page_group[0], page_group[-1]
            content = "\n\n".join(p.text for p in page_group)
            label = _page_label(page_group)
            page_range = (
                str(first.index) if first.index == last.index else f"{first.index}-{last.index}"
            )
            sec_key = section or f"page {label or first.index}"
            source = f"{src['slug']}#{sec_key} p.{label or first.index}"
            chunk_id = f"{src['slug']}|{sec_key}|{page_range}"
            content_hash = _sha256_bytes((content + "\n" + source).encode("utf-8"))
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=src["slug"],
                    source_id=src["source_id"],
                    title=src["title"],
                    edition=src["edition"],
                    section=sec_key,
                    page_pdf=first.index,
                    page_label=label,
                    page_range=page_range,
                    content=content,
                    tier=src["tier"],
                    audience=src["audience"],
                    rights_status=src["rights_status"],
                    vehicle_match=src["tum_vehicle_match"],
                    source=source,
                    sha256=content_hash,
                )
            )
    records.sort(key=lambda r: (r.doc_id, r.page_pdf, r.chunk_id))
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with open(corpus_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
    info = _corpus_info(manifest, records, out_dir)
    with open(corpus_dir / "corpus-info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return records


def _corpus_info(manifest: dict, records: list[ChunkRecord], out_dir: Path) -> dict:
    by_tier: dict[str, int] = {}
    by_audience: dict[str, int] = {}
    for rec in records:
        by_tier[rec.tier] = by_tier.get(rec.tier, 0) + 1
        by_audience[rec.audience] = by_audience.get(rec.audience, 0) + 1
    digests = "".join(r.sha256 for r in records)
    return {
        "primary_edition": manifest["primary_edition"],
        "chunk_count": len(records),
        "sha256_of_all": _sha256_bytes(digests.encode("utf-8")),
        "chunks_by_tier": dict(sorted(by_tier.items())),
        "chunks_by_audience": dict(sorted(by_audience.items())),
        "sources_ingested": [s["source_id"] for s in manifest["sources"] if s.get("ingested")],
        "sources_version_analysis_only": [
            s["source_id"] for s in manifest["sources"] if not s.get("ingested")
        ],
        "build_max_chars": 2200,
    }


def _page_shingles(text: str, n: int = 8) -> set[str]:
    norm = re.sub(r"[^a-z]+", "", text.lower())
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def _jac(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_dup_report(manifest: dict, evidence_dir: Path, out_dir: Path) -> dict:
    """Section (page) level near-duplicate matrix across parked Born editions.

    Editions marked ``ingested=false`` are never retrieved; this report only
    evidence the "conflicting/superseding editions" surface. Deterministic.
    """
    versions = [s for s in manifest["sources"] if s["role"] == "version-analysis-only"]
    by_id: dict[str, dict] = {}
    for src in versions:
        pdf_path = evidence_dir / src["evidence_file"]
        if not pdf_path.exists():
            raise CorpusError(f"missing evidence for {src['source_id']}")
        fallback = src["title"] if src["tier"] in ("family", "generic") else None
        pages = pages_from_pdf(pdf_path, src.get("page_count"), fallback)
        by_id[src["source_id"]] = {
            "edition": src["edition"],
            "page_count": src["page_count"],
            "shingles": [_page_shingles(p.text) for p in pages],
        }
    ids = list(by_id)
    matrix: dict[str, dict[str, float]] = {}
    dup_pages: dict[str, dict[str, int]] = {}
    for i in ids:
        matrix[i], dup_pages[i] = {}, {}
        ai = by_id[i]["shingles"]
        for j in ids:
            if i == j:
                matrix[i][j] = 1.0
                continue
            aj = by_id[j]["shingles"]
            sims = [max(_jac(s, t) for t in aj) if s else 0.0 for s in ai]
            dup_pages[i][j] = sum(1 for s in sims if s >= 0.85)
            matrix[i][j] = round(sum(sims) / len(sims) if sims else 0.0, 3)
    report = {
        "method": "page-level 8-gram Jaccard; max similarity per page; duplication = sim >= 0.85",
        "matrices_similarity": matrix,
        "matrices_dup_pages": dup_pages,
        "summary": [
            {
                "pair": f"{i} vs {j}",
                "mean_page_sim": matrix[i][j],
                "dup_pages_share": round(dup_pages[i][j] / max(by_id[i]["page_count"], 1), 3),
            }
            for i in ids
            for j in ids
            if i < j
        ],
    }
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "near-dup.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)