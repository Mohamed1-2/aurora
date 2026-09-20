"""AI-quality + LLM observability metrics shared across AURORA services.

These counters sum request outcomes (so they can also be surfaced per-applicaiton
in the gateway /stats and assistant /environment endpoints) and are exported as
Prometheus counters on every service's ``/metrics``.  They answer the platform
questions an ops / MLOps reviewer actually asks:

    grounding rate, refusal rate, fallback rate, hallucination proxies,
    cost per request, cache hit rate, latency distribution.
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    _AVAILABLE = True
except Exception:  # pragma: no cover
    _AVAILABLE = False


def _noop():  # pragma: no cover
    class _N:
        def labels(self, *_a, **_k):
            return self

        def inc(self, _n=1):  # noqa: B008
            return None

        def observe(self, _v):  # noqa: B008
            return None

        def set(self, _v):  # noqa: B008
            return None

    return _N()


class AiQualityMetrics:
    """AI/LLM quality counters: grounding, refusals, safety, costs, latency."""

    def __init__(self, service: str = "assistant") -> None:
        self.service = service
        if _AVAILABLE:
            self.requests = Counter(
                "aurora_ai_requests_total", "AI requests by outcome",
                ["service", "outcome"],
            )
            self.refusals = Counter(
                "aurora_ai_refusals_total", "Refusals (any tier)", ["service", "kind"],
            )
            self.grounded = Counter(
                "aurora_ai_grounded_total", "Answers grounded by source", ["service", "source"],
            )
            self.safety_blocks = Counter(
                "aurora_ai_safety_blocks_total", "Content-safety / policy blocks",
                ["service", "policy_level"],
            )
            self.fallbacks = Counter(
                "aurora_ai_fallbacks_total", "Provider fallbacks", ["service", "from_provider", "to_provider"],
            )
            self.cost_usd = Counter(
                "aurora_ai_cost_usd_total", "LLM cost in USD", ["service"],
            )
            self.latency_ms = Histogram(
                "aurora_ai_latency_ms", "AI request latency", ["service"],
                buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
            )
            self.cache_hit = Counter(
                "aurora_ai_cache_hits_total", "Cache hits", ["service"],
            )
            self.cache_miss = Counter(
                "aurora_ai_cache_misses_total", "Cache misses", ["service"],
            )
            self.tokens = Counter(
                "aurora_ai_tokens_total", "Tokens in/out by service", ["service", "direction"],
            )
        else:  # pragma: no cover
            for n in ("requests", "refusals", "grounded", "safety_blocks", "fallbacks",
                      "cost_usd", "latency_ms", "cache_hit", "cache_miss", "tokens"):
                setattr(self, n, _noop())

    def record(
        self,
        *,
        outcome: str = "ok",
        refused: bool = False,
        refusal_kind: str = "",
        grounded_sources: int = 0,
        safety_level: str = "read_only",
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        self.requests.labels(self.service, outcome).inc()
        self.latency_ms.labels(self.service).observe(latency_ms)
        if refused:
            self.refusals.labels(self.service, refusal_kind or "generic").inc()
        if grounded_sources > 0:
            self.grounded.labels(self.service, "telemetry").inc(grounded_sources)
        if cost_usd:
            self.cost_usd.labels(self.service).inc(cost_usd)
        if cache_hit:
            self.cache_hit.labels(self.service).inc()
        else:
            self.cache_miss.labels(self.service).inc()
        if tokens_in:
            self.tokens.labels(self.service, "in").inc(tokens_in)
        if tokens_out:
            self.tokens.labels(self.service, "out").inc(tokens_out)


def ai_quality_metrics(service: str = "assistant") -> AiQualityMetrics:
    return AiQualityMetrics(service)
