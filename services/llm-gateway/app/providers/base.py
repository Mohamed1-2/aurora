"""Provider interface + result types for the LLM gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vehicle_ai.schemas import GatewayRequest


@dataclass
class ProviderResult:
    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    region: str | None = None
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1000.0) * self.cost_per_1k_in + (
            self.completion_tokens / 1000.0
        ) * self.cost_per_1k_out


class ProviderError(RuntimeError):
    """Raised when a provider call fails so the router can fail over."""


class BaseProvider:
    """Common surface every provider must present to the router."""

    name = "base"
    models: list[str] = []

    def is_configured(self) -> bool:
        return False

    async def complete(self, req: GatewayRequest) -> ProviderResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "models": self.models, "configured": self.is_configured()}


def extract_grounding(req: GatewayRequest) -> dict[str, Any]:
    """Normalise ``req.grounding`` for providers that consume JSON context."""
    g = req.grounding
    if g is None:
        return {"telemetry": {}, "documents": [], "events": []}
    return {
        "telemetry": g.telemetry or {},
        "documents": [d.model_dump() for d in g.documents],
        "events": g.events or [],
    }