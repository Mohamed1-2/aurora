"""AURORA LLM Gateway — the single inference entry point.

Routes assistant traffic to multiple model providers with circuit breakers,
ordered failover, semantic caching and per-model spend accounting — the piece
that keeps a fleet of cars online when a model region degrades.
"""

from __future__ import annotations

import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from vehicle_ai.logging_conf import configure_logging, get_logger
from vehicle_ai.metrics import MetricsRegistry, metrics_available, prometheus_asgi_app
from vehicle_ai.schemas import GatewayRequest, GatewayResponse
from vehicle_ai.version import PROMPT_VERSION, __version__

from app.config import settings
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.peft import PEFTProvider
from app.router import LlmRouter

configure_logging("llm-gateway", settings.log_level)
logger = get_logger("aurora.gateway")
metrics = MetricsRegistry()

_providers: dict[str, object] = {"mock": MockProvider()}

if settings.local_enabled:
    _providers["local"] = OpenAICompatProvider(
        "local", api_key="ollama", base_url=settings.local_base_url,
        model=settings.local_model, region="on-device", timeout_seconds=35.0,
    )

if settings.peft_adapter_dir:
    _providers["peft"] = PEFTProvider(
        settings.peft_adapter_dir,
        base_model=settings.peft_base_model,
        use_lora=settings.peft_use_lora,
        max_new_tokens=settings.peft_max_new_tokens,
    )
    # Base-vs-LoRA side-by-side for the v2 harness: same weights family, LoRA off.
    _providers["peft-base"] = PEFTProvider(
        settings.peft_adapter_dir,
        base_model=settings.peft_base_model,
        use_lora=False,
        max_new_tokens=settings.peft_max_new_tokens,
    )

if settings.openai_base_url and settings.openai_api_key:
    _providers["openai"] = OpenAICompatProvider(
        "openai",
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        region=None,
        cost_per_1k_in=0.00015,
        cost_per_1k_out=0.00060,
    )

if settings.azure_openai_endpoint and settings.azure_openai_api_key:
    _providers["azure-openai"] = OpenAICompatProvider(
        "azure-openai",
        api_key=settings.azure_openai_api_key,
        base_url=settings.azure_openai_endpoint,
        model=settings.azure_openai_deployment,
        region="azure",
        cost_per_1k_in=0.00015,
        cost_per_1k_out=0.00060,
        azure_deployment=settings.azure_openai_deployment,
        azure_api_version=settings.azure_openai_api_version,
    )

_priority = [p.strip() for p in settings.provider_priority.split(",") if p.strip() and p.strip() in _providers]
if settings.local_enabled:
    _priority = ["local"] + [p for p in _priority if p != "local"]

_weights = {
    k: int(v)
    for k, v in (pair.split(":") for pair in settings.provider_weights.split(","))
    if k in _providers
}

router = LlmRouter(
    providers=_providers,  # type: ignore[arg-type]
    priority=_priority,
    weights=_weights,
    cache_ttl=settings.cache_ttl_seconds,
    cache_max=settings.cache_max_entries,
    breaker_threshold=settings.circuitbreaker_failures,
    breaker_cooldown=settings.circuitbreaker_cooldown_s,
)

app = FastAPI(
    title="AURORA LLM Gateway",
    version=__version__,
    description="Multi-provider inference routing, failover, caching and cost control.",
)


class CompleteRequest(GatewayRequest):
    pass


from app.operations_demo import router as operations_router
app.include_router(operations_router)


@app.post("/v1/complete", response_model=GatewayResponse)
async def complete(req: CompleteRequest):
    resp = await router.route(req)
    metrics.requests_total.labels("llm-gateway", "/v1/complete").inc()
    metrics.request_latency_ms.labels("llm-gateway", "/v1/complete").observe(resp.latency_ms)
    return resp


@app.post("/v1/debug/prompt")
async def debug_prompt(req: CompleteRequest):
    """Return the exact composed prompt for a pinned provider — no inference.

    The v2 harness (compare_v2.py) uses this to prove byte-for-byte that
    ``peft`` and ``peft-base`` receive identical prompts (only the adapter
    differs), and to diff the rendered prompt against the canonical
    ``context_prompt`` output. Works on GPU-less machines because prompt
    assembly never touches the model weights.
    """
    if not req.provider:
        return {"error": "provider required", "provider": None, "prompt": None}
    p = router.providers.get(req.provider)
    if p is None:
        return {"error": f"unknown provider {req.provider!r}", "provider": req.provider, "prompt": None}
    renderer = getattr(p, "render_prompt", None)
    if renderer is None:
        return {"error": f"{req.provider} has no prompt renderer", "provider": req.provider, "prompt": None}
    return {"provider": req.provider, "prompt": renderer(req)}


@app.post("/v1/embed")
async def embed(req: dict):
    """Embedding endpoint (hybrid-RAG path). Requires an embedding provider."""
    return {"error": "embedding provider not configured", "note": "configure via AURORA_GATEWAY_EMBED_BASE_URL"}


@app.get("/providers")
def providers():
    return {
        name: {"configured": p.is_configured(), "models": p.models, "region": getattr(p, "region", None)}
        for name, p in router.providers.items()
    }


@app.get("/routes")
def routes():
    return {
        "priority_chain": _priority,
        "weights": _weights,
        "health": router.health(),
    }


@app.get("/stats")
def stats():
    return {
        "cache_entries": len(router.cache),
        "cache_ttl_s": router.cache_ttl,
        "spend_usd_by_provider": router.spend_usd,
        "calls_by_provider": router.calls,
        "refresh_stats_at": time.time(),
    }


@app.get("/healthz")
def healthz():
    healthy_chain = [p for p in _priority if router.health()[p]["circuit"] != "open"]
    return {"status": "ok", "service": "llm-gateway", "providers": len(_providers), "healthy_chain": healthy_chain}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "prompt_version": PROMPT_VERSION}


@app.get("/metrics")
def metrics_endpoint():
    if not metrics_available():
        return Response("prometheus disabled", media_type="text/plain")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, log_config=None)