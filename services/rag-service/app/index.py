"""RAG: provenanced corpus indexing + tier/audience-aware BM25 retrieval.

The index is deterministic: chunking and scoring are pure functions of the
corpus, so the index hash pins the exact retrieval behaviour an evaluation
run was scored against (``aurora.rag_index_version``).

v4 changes over v3:
* Corpus chunks carry full provenance (source edition, PDF page, tier,
  audience, rights status, vehicle-match status) instead of synthetic docs.
* ``retrieve()`` applies an authority-tier multiplier on top of BM25
  (manual > appendix > vehicle_specific > family > generic) — a tie-breaker,
  not a dominant signal, so a generic doc with real term matches still wins
  over a manual page that merely brushes the query.
* An audience gate keeps responder material (rescue sheets / ERG guides) out
  of default driver retrieval.
* A small curated query-expansion table repairs lexicon divergence
  (tyre/tire, soc/state of charge, km-h units) and headline synonyms.
* Query-side stopword filtering removes function words ("What is my range
  now?" no longer matches on repeated "is") without touching chunk tokens.
* ``layered()`` builds a composite index: the provenanced corpus is the
  primary authority; the committed knowledge-pack sits below it as the
  assistant's coaching fallback (knowledge tier, provenance-carrying chunks).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from vehicle_ai.schemas import RAGChunk

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words add only lexical noise to BM25 matching in natural-language
# queries ("What is my estimated range now?" must not win a page because it
# repeats "is" fourteen times). Removed from the *query* side; chunk tokens
# stay untouched so document statistics are stable and query-side removal
# simply means those tokens can no longer match.
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "of", "to", "for", "with",
    "in", "on", "at", "from", "by", "about", "as", "into", "over", "under",
    "is", "are", "was", "were", "be", "being", "been", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "should", "could", "can",
    "may", "might", "shall", "my", "your", "our", "their", "his", "her",
    "its", "i", "you", "we", "they", "it", "me", "mine", "yours", "what",
    "which", "who", "when", "where", "why", "how", "right", "now", "please",
    "this", "that", "these", "those", "doing", "okay", "well", "sure",
})

# Query-only stop-application (cached so hot-path calls are cheap).
def _query_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]

AUDIENCES = ("owner", "responder", "all")

# Curated query-expansion: synonym token -> extra tokens to also match.
# Conservative on purpose; heavy expansion dilutes precision.
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "tyre": ("tire",),
    "tire": ("tyre",),
    "tyres": ("tires",),
    "tires": ("tyres",),
    "soc": ("charge", "charged"),
    "high-voltage": ("hv",),
    "hv": ("high", "voltage"),
    "regenerative": ("regen",),
    "regen": ("regenerative", "braking"),
    "ev": ("electric", "vehicle"),
    "winter": ("cold",),
    "cold": ("winter",),
    "precondition": ("pre-condition", "preheating", "climatisation", "stationary", "departure"),
    "preheating": ("precondition", "climatisation"),
    "preheat": ("preheating", "precondition", "heat", "stationary"),
    "climatisation": ("precondition", "preheating"),
    "stationary": ("precondition",),
    "cool": ("cooling", "climate", "ventilation"),
    "cabin": ("interior",),
    "air-conditioning": ("aircon",),
    "change": ("changing", "replacing", "replace"),
    "changing": ("change", "replacing"),
    "replace": ("changing", "replacement"),
    "serviced": ("service", "services", "maintenance"),
    "interval": ("service", "inspection"),
    "warning": ("lamp", "lamps", "indicator", "indicators"),
    # No "dashboard" expansion: instrument/cluster/display are page-ubiquitous
    # and drowned the warning-lamp catalogue in generic cockpit matches; the
    # literal term already matches wherever the manual uses it.
}

TIER_MULTIPLIER = {
    "manual": 1.7,
    "appendix": 1.6,
    "vehicle_specific": 1.35,
    "family": 1.15,
    "generic": 1.0,
    "knowledge": 0.9,  # committed assistant coaching layer, lowest authority
}

DOC_TITLES: dict[str, str] = {
    "01-vehicle-overview.md": "Vehicle Overview",
    "02-battery-and-charging.md": "Battery and Charging",
    "03-range-optimization.md": "Range Optimization",
    "04-tires-and-wheels.md": "Tyres and Wheels",
    "05-diagnostics-and-dtc.md": "Diagnostics and Trouble Codes",
    "06-emergency-and-safety.md": "Emergency and Safety",
    "07-winter-and-weather.md": "Winter and Weather",
    "08-maintenance-schedule.md": "Maintenance Schedule",
    "09-owners-manual-extended.md": "Owners Manual: Extended",
}

# Explicit, honest provenance for the committed assistant knowledge-pack: it is
# internal curated guidance, NOT manufacturer documentation, carries no source
# file/page, and must stay beneath the verified manual tier in retrieval.
KNOWLEDGE_RIGHTS = "aurora-curated assistant guide — not manufacturer documentation"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def expand_query(query: str) -> list[str]:
    """Return all matching terms for a query (stopwords removed), incl. synonyms."""
    qt = list(dict.fromkeys(_query_tokens(query)))
    for i, t in enumerate(list(qt)):
        for extra in EXPANSIONS.get(t, ()):
            if extra not in STOPWORDS and extra not in qt:
                qt.append(extra)
    return qt


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    content: str
    tokens: list[str] = field(default_factory=list, repr=False)
    # Provenance (corpus-backed; empty for legacy markdown sources).
    source_id: str = ""
    edition: str = ""
    page_pdf: int | None = None
    page_label: str | None = None
    page_range: str = ""
    tier: str = ""
    audience: str = ""
    rights_status: str = ""
    vehicle_match: str = ""
    source: str = ""

    @property
    def provenance(self) -> dict:
        return {
            "source_id": self.source_id,
            "edition": self.edition,
            "page_pdf": self.page_pdf,
            "page_label": self.page_label,
            "page_range": self.page_range,
            "tier": self.tier,
            "audience": self.audience,
            "rights_status": self.rights_status,
            "vehicle_match": self.vehicle_match,
        }


@dataclass
class IndexStats:
    n_docs: int
    n_chunks: int
    n_terms: int
    avg_chunk_len: float
    hash: str
    corpus_sha256: str = ""
    n_by_tier: dict[str, int] | None = None
    n_by_audience: dict[str, int] | None = None
    docs: list[str] | None = None


class RagIndex:
    """BM25 index over the provenanced corpus. k1=1.5, b=0.75."""

    def __init__(self, sources: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.corpus_sha256 = ""
        for doc_id, text in sources.items():
            for section, body in _split_doc(doc_id, text):
                self.chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=DOC_TITLES.get(doc_id, doc_id),
                        section=section,
                        content=body,
                        tokens=tokenize(f"{section} {body}"),
                        source=f"{doc_id}#{section}",
                    )
                )
        self._finalize()

    @classmethod
    def from_corpus(cls, corpus_file: str, k1: float = 1.5, b: float = 0.75) -> RagIndex:
        self = cls.__new__(cls)
        self.k1, self.b = k1, b
        self.chunks = _load_corpus_records(corpus_file)
        with open(_corpus_info_path(corpus_file), encoding="utf-8") as fh:
            info = json.load(fh)
        self.corpus_sha256 = info.get("sha256_of_all", "")
        self._finalize()
        return self

    @classmethod
    def layered(
        cls,
        corpus_file: str | None,
        kb_dir: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> RagIndex:
        """Composite index: the provenanced corpus is the primary authority, the
        committed assistant knowledge-pack (markdown) is a lowest-authority
        coaching layer. The manual lands wherever it genuinely answers;
        coaching knowledge the manual never states (aero/regen guidance,
        DTC semantics, service intervals) falls back to the knowledge layer.
        """
        self = cls.__new__(cls)
        self.k1, self.b = k1, b
        self.chunks = []
        self.corpus_sha256 = ""
        corpus_path = Path(corpus_file) if corpus_file else None
        if corpus_path and corpus_path.is_file():
            self.chunks = _load_corpus_records(corpus_file)
            info = json.loads(Path(_corpus_info_path(corpus_file)).read_text(encoding="utf-8"))
            self.corpus_sha256 = info.get("sha256_of_all", "")
        if kb_dir and Path(kb_dir).is_dir():
            for doc_id, text in load_kb(kb_dir).items():
                for section, body in _split_doc(doc_id, text):
                    self.chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            title=DOC_TITLES.get(doc_id, doc_id),
                            section=section,
                            content=body,
                            tokens=tokenize(f"{section} {body}"),
                            tier="knowledge",
                            audience="owner",
                            rights_status=KNOWLEDGE_RIGHTS,
                            vehicle_match="unproven",
                            source=f"{doc_id}#{section}",
                        )
                    )
        self._finalize()
        return self

    def _finalize(self) -> None:
        self.n_docs = len(self.chunks)
        self.avg_len = statistics.fmean(len(c.tokens) for c in self.chunks) or 1.0
        df: dict[str, int] = {}
        for c in self.chunks:
            for t in set(c.tokens):
                df[t] = df.get(t, 0) + 1
        self.idf: dict[str, float] = {}
        for t, d in df.items():
            self.idf[t] = math.log(1 + (self.n_docs - d + 0.5) / (d + 0.5))
        by_tier: dict[str, int] = {}
        by_aud: dict[str, int] = {}
        for c in self.chunks:
            if c.tier:
                by_tier[c.tier] = by_tier.get(c.tier, 0) + 1
            if c.audience:
                by_aud[c.audience] = by_aud.get(c.audience, 0) + 1
        self.stats = IndexStats(
            n_docs=len({c.doc_id for c in self.chunks}),
            n_chunks=len(self.chunks),
            n_terms=len(self.idf),
            avg_chunk_len=round(self.avg_len, 2),
            hash=_index_hash(self.chunks),
            corpus_sha256=self.corpus_sha256,
            n_by_tier=by_tier or None,
            n_by_audience=by_aud or None,
            docs=sorted({c.doc_id for c in self.chunks}),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_weights: dict[str, float] | None = None,
        audience: str = "owner",
    ) -> list[RAGChunk]:
        doc_weights = doc_weights or {}
        if audience not in AUDIENCES:
            audience = "owner"
        qt = expand_query(query)
        if not qt:
            return []
        qt_set = set(qt)
        scores: list[tuple[float, Chunk]] = []
        for c in self.chunks:
            if not _audience_allowed(c.audience, audience):
                continue
            tf: dict[str, int] = {}

            def _add(ts: list[str], weight: float) -> None:
                local: dict[str, int] = {}
                for t in ts:
                    local[t] = local.get(t, 0) + 1
                for t, n in local.items():
                    tf[t] = tf.get(t, 0) + n  # type: ignore[assignment]

            _add(c.tokens, 1.0)
            score = 0.0
            dl = len(c.tokens) or 1
            for t in qt:
                if t not in self.idf:
                    continue
                n = tf.get(t, 0)
                if n == 0:
                    continue
                idf = self.idf[t]
                score += idf * (n * (self.k1 + 1)) / (n + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
            if c.section and any(t in c.section.lower() for t in qt_set):
                score += 0.8 * len(qt_set & set(tokenize(c.section)))
            multiplier = TIER_MULTIPLIER.get(c.tier, 1.0) if c.tier else 1.0
            score *= multiplier
            page_boost = 1.03 if c.page_pdf else 1.0
            score *= page_boost * doc_weights.get(c.doc_id, 1.0)
            if score > 0:
                scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        out: list[RAGChunk] = []
        for score, c in scores[:top_k]:
            # Provenance is always emitted. Corpus chunks carry full origin
            # (source edition, pages, tier); knowledge-pack chunks carry an
            # empty origin plus a rights marker so they can never be mistaken
            # for manufacturer documentation.
            provenance = c.provenance
            out.append(
                RAGChunk(
                    doc_id=c.doc_id,
                    section=c.section,
                    title=c.title,
                    content=c.content[:700],
                    score=round(score, 4),
                    source=c.source or f"{c.doc_id}#{c.section}",
                    **provenance,
                )
            )
        return out


def _audience_allowed(chunk_audience: str, request: str) -> bool:
    if not chunk_audience:
        return True  # legacy chunks are owner-facing
    if request == "all":
        return True
    if request == "responder":
        return chunk_audience in ("owner", "responder")
    return chunk_audience == "owner"  # default owner gate excludes responders


def _load_corpus_records(corpus_file: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(corpus_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks.append(
                Chunk(
                    doc_id=rec["doc_id"],
                    title=rec["title"],
                    section=rec["section"],
                    content=rec["content"],
                    tokens=tokenize(f"{rec['section']} {rec['content']}"),
                    source_id=rec["source_id"],
                    edition=rec["edition"],
                    page_pdf=rec.get("page_pdf"),
                    page_label=rec.get("page_label"),
                    page_range=rec.get("page_range", ""),
                    tier=rec["tier"],
                    audience=rec["audience"],
                    rights_status=rec["rights_status"],
                    vehicle_match=rec["vehicle_match"],
                    source=rec["source"],
                )
            )
    return chunks


def _corpus_info_path(corpus_file: str) -> str:
    p = Path(corpus_file)
    return str(p.with_name("corpus-info.json"))


def _split_doc(doc_id: str, text: str) -> list[tuple[str, str]]:
    """Split a Markdown KB doc on its H2 (##) sections (legacy path)."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_section, current_body = "overview", []
    for line in lines:
        if line.startswith("## "):
            if current_body:
                sections.append((current_section, current_body))
            current_section = line[3:].strip()
            current_body = []
        elif line.startswith("# "):
            continue
        else:
            if line.strip():
                current_body.append(line.strip())
    if current_body:
        sections.append((current_section, current_body))
    return [(s, " ".join(body)) for s, body in sections if body]


def _index_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.doc_id}|{c.section}|{c.page_pdf}|{c.tier}|{c.audience}|{c.content}".encode())
    return h.hexdigest()[:16]


def load_kb(directory: str) -> dict[str, str]:
    """Read all ``*.md`` files in a directory keyed by filename (legacy)."""
    sources: dict[str, str] = {}
    for p in sorted(Path(directory).glob("*.md")):
        sources[p.name] = p.read_text(encoding="utf-8")
    return sources


def resolve_layers(corpus_file: str | None, kb_dir: str | None, service_root: Path) -> tuple[str | None, str | None]:
    """Resolve the retrieval layers present at deploy time.

    Returns ``(corpus_path, kb_dir)``; a None corpus means the provenanced
    layer is not available (CI / fresh checkout), a None kb means the
    assistant coaching layer is absent. Preference order for the corpus:
    explicit setting > ``<repo>/artifacts/rag/corpus/chunks.jsonl``.
    """
    corpus_candidates: list[str] = []
    if corpus_file:
        corpus_candidates.append(corpus_file)
    corpus_candidates.append(str(service_root / "artifacts/rag/corpus/chunks.jsonl"))
    resolved_corpus = next((c for c in corpus_candidates if Path(c).is_file()), None)
    if not resolved_corpus and kb_dir and Path(kb_dir).is_dir():
        resolved_corpus = None  # knowledge layer only
    kb = kb_dir if (kb_dir and Path(kb_dir).is_dir()) else service_root / "services/rag-service/knowledge_base"
    resolved_kb = str(kb) if Path(kb).is_dir() else None
    return (resolved_corpus, resolved_kb) if (resolved_corpus or resolved_kb) else (None, None)