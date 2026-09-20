"""Inference routing: availability, failover, cache and cost.

Architecture notes:
* Providers run in a config-driven *priority chain*. A request always starts
  at the front of the chain and falls back along it until a provider succeeds.
* Each (provider, model) pair has a **circuit breaker**: after N consecutive
  failures it is tripped open for a cooldown window and requests skip it
  immediately (fail fast), which protects tail p99 latency.
* Responses are cached by content hash (per prompt version) so identical
  questions never spend tokens twice within the TTL window.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.normalizer import normalizer_for
from vehicle_ai.schemas import GatewayRequest, GatewayResponse, SafetyVerdict

from .providers.base import BaseProvider, ProviderError
from .safety import deterministic_gate, evaluate_safety

logger = get_logger("aurora.gateway.router")
_ai_q = None


def _ai_metrics():
    global _ai_q
    if _ai_q is None:
        from vehicle_ai.ai_metrics import ai_quality_metrics

        _ai_q = ai_quality_metrics("gateway")
    return _ai_q


@dataclass
class RouteDecision:
    provider: str
    model: str
    fallback_chain: list[str] = field(default_factory=list)
    cache_hit: bool = False


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: int) -> None:
        self.threshold = threshold
        self.cooldown = cooldown_s
        self.failures = 0
        self.open_until = 0.0
        self._half_open_test = False

    @property
    def state(self) -> str:
        if time.time() < self.open_until:
            return "open"
        if self._half_open_test:
            return "half-open"
        return "closed"

    def allow(self) -> bool:
        if time.time() < self.open_until:
            return False
        if self._half_open_test:
            return False  # let exactly one probe through; handled by caller
        return True

    def acquire_probe(self) -> bool:
        if self._half_open_test:
            return False
        self._half_open_test = True
        return True

    def success(self) -> None:
        self.failures = 0
        self.open_until = 0.0
        self._half_open_test = False

    def failure(self) -> None:
        self._half_open_test = False
        self.failures += 1
        if self.failures >= self.threshold and time.time() >= self.open_until:
            self.open_until = time.time() + self.cooldown
            logger.warning("circuit breaker opened", extra={"payload": {"failures": self.failures}})


class LlmRouter:
    def __init__(
        self,
        providers: dict[str, BaseProvider],
        priority: list[str],
        weights: dict[str, int],
        cache_ttl: int,
        cache_max: int,
        breaker_threshold: int,
        breaker_cooldown: int,
    ) -> None:
        self.providers = providers
        self.priority = priority
        self.weights = weights
        self.breakers: dict[str, CircuitBreaker] = {}
        self.cache: OrderedDict[str, tuple[float, GatewayResponse]] = OrderedDict()
        self.cache_ttl = cache_ttl
        self.cache_max = cache_max
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self.spend_usd: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    # -- circuit breakers ----------------------------------------------------

    def _breaker_key(self, provider: str, model: str) -> str:
        return f"{provider}/{model}"

    def _breaker(self, key: str) -> CircuitBreaker:
        if key not in self.breakers:
            self.breakers[key] = CircuitBreaker(self.breaker_threshold, self.breaker_cooldown)
        return self.breakers[key]

    # -- priority / pinning ---------------------------------------------------

    def chain_for(self, req: GatewayRequest) -> list[tuple[str, BaseProvider]]:
        pinned = []
        if req.provider and req.provider in self.providers:
            pinned.append((req.provider, self.providers[req.provider]))
        chain = [(name, self.providers[name]) for name in self.priority if name in self.providers]
        seen = {p for p, _ in pinned}
        return pinned + [(n, p) for n, p in chain if n not in seen]

    # -- cache ----------------------------------------------------------------

    def _cache_key(self, req: GatewayRequest) -> str | None:
        if req.cache_key:
            return f"ck:{req.cache_key}"
        if not req.messages:
            return None
        base = "|".join(f"{m.role}:{m.content}" for m in req.messages)
        g = req.grounding
        if g:
            base += "|tel:" + hashlib.md5(repr(sorted(g.telemetry.items())).encode()).hexdigest()[:12] if g.telemetry else ""
            base += "|doc:" + ",".join(d.source for d in g.documents) if g.documents else ""
        # A pinned provider/model is part of the cache identity: two providers
        # must never serve each other's cached response for an identical prompt.
        base += f"|p:{req.provider or ''}"
        base += f"|m:{req.model or ''}"
        base += f"|generation:{req.temperature}:{req.max_tokens}"
        return "q:" + hashlib.sha256(base.encode()).hexdigest()

    def _evict_expired(self, now: float) -> None:
        while self.cache:
            k, (ts, _) = next(iter(self.cache.items()))
            if now - ts > self.cache_ttl:
                self.cache.pop(k)
            else:
                break

    # -- cost ledger ----------------------------------------------------------

    def record_spend(self, provider: str, usd: float) -> None:
        self.spend_usd[provider] = self.spend_usd.get(provider, 0.0) + usd
        self.calls[provider] = self.calls.get(provider, 0) + 1

    # -- main entry -----------------------------------------------------------

    @staticmethod
    def _canonical_dict(req: GatewayRequest) -> dict:
        g = req.grounding
        telemetry = g.telemetry if g else {}
        source = req.metadata.get("telemetry_source") or "orionx_live"
        charging = bool(req.metadata.get("telemetry_charging"))
        return normalizer_for(source, charging=charging)(telemetry).model_dump()

    async def route(self, req: GatewayRequest) -> GatewayResponse:
        start = time.perf_counter()
        # Deterministic pre-LLM gate: prohibited vehicle-control and missing-data
        # requests never reach a model that cannot be trusted to refuse them.
        # Model-level evaluation (compare_v2.py --evaluate-model) opts out so the
        # raw Base/LoRA refusal behaviour can be measured and improved; production
        # traffic always passes through the gate.
        evaluate_model = bool(req.metadata.get("evaluate_model"))
        gate_user = req.messages[-1].content if req.messages else ""
        decision = deterministic_gate(gate_user, self._canonical_dict(req))
        if not decision.allowed and not evaluate_model:
            resp = GatewayResponse(
                trace_id=req.trace_id,
                provider="policy-gate",
                model="",
                content=decision.content,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                safety=SafetyVerdict(
                    approved=False,
                    policy="vehicle-assistant-default",
                    blocked_reasons=[decision.reason],
                    score=0.0,
                ),
                refused=True,
            )
            try:
                _ai_metrics().record(
                    outcome="blocked",
                    refused=True,
                    refusal_kind="policy",
                    latency_ms=resp.latency_ms,
                    cost_usd=0.0,
                    cache_hit=False,
                    safety_level="prohibited",
                )
            except Exception:  # noqa: BLE001
                pass
            return resp

        ck = self._cache_key(req)
        now = time.time()
        if ck and ck in self.cache:
            ts, resp = self.cache[ck]
            if resp.provider and now - ts <= self.cache_ttl:
                resp = resp.model_copy(update={"trace_id": req.trace_id, "cache_hit": True})
                self.record_spend(resp.provider, 0.0)
                try:
                    _ai_metrics().record(cache_hit=True, latency_ms=(time.perf_counter() - start) * 1000)
                except Exception:  # noqa: BLE001
                    pass
                return resp
            self.cache.pop(ck)

        chain = self.chain_for(req)
        fallbacks: list[str] = []
        last_error = ""
        for name, provider in chain:
            key = self._breaker_key(name, provider.models[0] if provider.models else name)
            br = self._breaker(key)
            if br.state == "open":
                fallbacks.append(f"{name} (circuit-open)")
                continue
            try:
                result = await provider.complete(req)
            except ProviderError as exc:
                last_error = str(exc)
                br.failure()
                fallbacks.append(f"{name} -> error")
                logger.error(f"provider failed", extra={"payload": {"provider": name, "error": last_error}})
                continue
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                br.failure()
                fallbacks.append(f"{name} -> error")
                continue

            br.success()
            if fallbacks:
                try:
                    _ai_metrics().fallbacks.labels("gateway", fallbacks[0].split()[0], name).inc()
                except Exception:  # noqa: BLE001
                    pass
            verdict = evaluate_safety(
                req.messages[-1].content if req.messages else "",
                result.content,
            )
            usd = result.cost_usd
            self.record_spend(name, usd)
            resp = GatewayResponse(
                trace_id=req.trace_id,
                provider=name,
                model=result.model,
                region=result.region,
                content=result.content,
                reasoning=result.reasoning,
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                cost_usd=round(usd, 6),
                fallback_chain=fallbacks,
                cache_hit=False,
                safety=verdict,
                refused=not verdict.approved,
            )
            if ck:
                self.cache[ck] = (now, resp)
                if len(self.cache) > self.cache_max:
                    self._evict_expired(now)
            try:
                _ai_metrics().record(
                    outcome="blocked" if resp.refused else "ok",
                    refused=resp.refused,
                    refusal_kind="safety" if resp.refused else "",
                    latency_ms=resp.latency_ms,
                    cost_usd=usd,
                    cache_hit=False,
                    tokens_in=result.prompt_tokens,
                    tokens_out=result.completion_tokens,
                    safety_level="prohibited" if resp.refused else "read_only",
                )
            except Exception:  # noqa: BLE001
                pass
            return resp

        resp = GatewayResponse(
            trace_id=req.trace_id,
            provider="none",
            model="",
            content="",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            fallback_chain=fallbacks,
            safety=SafetyVerdict(approved=False, blocked_reasons=[f"all providers failed: {last_error}"]),
            refused=True,
        )
        # don't cache a total failure for the full TTL — allow a fast retry
        if ck:
            self.cache[ck] = (now - self.cache_ttl + 5, resp)
        return resp

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, p in self.providers.items():
            models = p.models or ["–"]
            key = self._breaker_key(name, models[0])
            out[name] = {
                "configured": p.is_configured(),
                "models": models,
                "circuit": self._breaker(key).state,
                "spend_usd": round(self.spend_usd.get(name, 0.0), 4),
                "calls": self.calls.get(name, 0),
            }
        return out


def resp_latency_to_zero(resp: GatewayResponse) -> float:
    return float(resp.latency_ms)