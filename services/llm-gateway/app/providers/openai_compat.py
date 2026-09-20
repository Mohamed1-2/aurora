"""OpenAI-compatible provider — works with Azure OpenAI and any OpenAI SDK.

Keenly important in production: the rest of the gateway treats every model
the same, so swapping GPT-4o for a locally hosted vLLM endpoint is a config
change, not a code change.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.schemas import GatewayRequest, Message

from .base import BaseProvider, ProviderError, ProviderResult

logger = get_logger("aurora.gateway.openai")


class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        region: str | None = None,
        cost_per_1k_in: float = 0.0,
        cost_per_1k_out: float = 0.0,
        azure_deployment: str | None = None,
        azure_api_version: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.models = [model]
        self.region = region
        self._cost_in = cost_per_1k_in
        self._cost_out = cost_per_1k_out
        self._azure_deployment = azure_deployment
        self._azure_api_version = azure_api_version
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    def is_configured(self) -> bool:
        return bool(self._api_key and self._base_url)

    def _url(self) -> str:
        if self._azure_deployment:
            return (
                f"{self._base_url}/openai/deployments/{self._azure_deployment}"
                f"/chat/completions?api-version={self._azure_api_version}"
            )
        return f"{self._base_url}/chat/completions"

    async def complete(self, req: GatewayRequest) -> ProviderResult:
        if not self.is_configured():
            raise ProviderError(f"{self.name} not configured")

        start = time.perf_counter()
        payload = {
            "model": self.models[0],
            "messages": [_m(m) for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = await self._client.post(self._url(), json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}")
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} connection error: {exc}")

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        latency_ms = (time.perf_counter() - start) * 1000
        return ProviderResult(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            model=self.models[0],
            region=self.region,
            cost_per_1k_in=self._cost_in,
            cost_per_1k_out=self._cost_out,
        )

    async def close(self) -> None:
        await self._client.aclose()


def _m(m: Message) -> dict[str, str]:
    return {"role": m.role, "content": m.content}