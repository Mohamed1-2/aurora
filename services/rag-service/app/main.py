"""RAG retrieval service — grounds assistant answers in the KB."""

from __future__ import annotations

import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vehicle_ai.logging_conf import configure_logging, get_logger
from vehicle_ai.metrics import MetricsRegistry
from vehicle_ai.version import RAG_INDEX_VERSION, __version__

from app.config import settings
from app.index import RagIndex, resolve_layers

configure_logging("rag-service", settings.log_level)
logger = get_logger("aurora.rag")
metrics = MetricsRegistry()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_corpus_path, _kb_path = resolve_layers(settings.corpus_file, settings.kb_dir, _REPO_ROOT)
index = RagIndex.layered(_corpus_path, _kb_path)
if _corpus_path:
    logger.info("rag index: corpus %s + knowledge layer %s (%d chunks)",
                _corpus_path, _kb_path or "-", index.stats.n_chunks)
else:
    logger.warning("rag index: knowledge layer only %s (%d chunks)", _kb_path, index.stats.n_chunks)

app = FastAPI(title="AURORA RAG", version=__version__)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None
    doc_weights: dict[str, float] | None = None
    audience: str = "owner"


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "rag-service"}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "chunks": index.stats.n_chunks}


@app.get("/index-info")
@metrics.observe("rag-service", "/index-info")
def index_info():
    s = index.stats
    return {
        "rag_index_version": f"{RAG_INDEX_VERSION}-{s.hash}",
        "n_chunks": s.n_chunks,
        "n_terms": s.n_terms,
        "avg_chunk_len": s.avg_chunk_len,
        "corpus_sha256": s.corpus_sha256 or None,
        "chunks_by_tier": s.n_by_tier or {},
        "chunks_by_audience": s.n_by_audience or {},
        "documents": s.docs or [],
    }


@app.post("/retrieve")
@metrics.observe("rag-service", "/retrieve")
def retrieve(req: RetrieveRequest):
    start = time.perf_counter()
    results = index.retrieve(
        req.query,
        top_k=req.top_k or settings.top_k,
        doc_weights=req.doc_weights,
        audience=req.audience,
    )
    return {
        "query": req.query,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "top_k": results,
        "index_version": f"{RAG_INDEX_VERSION}-{index.stats.hash}",
    }


@app.get("/metrics")
def metrics_endpoint():
    from fastapi.responses import Response
    from vehicle_ai.metrics import metrics_available, prometheus_asgi_app

    if not metrics_available():
        return Response("prometheus disabled", media_type="text/plain")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_config=None)