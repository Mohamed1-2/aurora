"""PEFT (LoRA) provider — serves the validated AURORA v2 adapter locally.

This provider mirrors the *exact* inference path that produced the locked
model-level evidence (adapter2 on SmolLM2-135M-Instruct):

* prompt format ``<|im_start|>system|user|assistant`` exactly as trained,
* telemetry rendered through ``vehicle_ai.context_prompt`` (the same module the
  v2 harness uses), so serve-time prompts == train-time prompts byte-for-byte,
* decoder settings copied from the evaluate notebook: ``max_new_tokens=64``,
  ``do_sample=False``.

Only the missing-data and refusal behaviour is inherited from the base provider
surface. The provider is *always* importable: torch/transformers/peft and the
adapter directory are imported lazily in :meth:`_load`, so the service boots and
reports ``configured=False`` on machines without a GPU/weights instead of
crashing — the exact "no torch/transformers/peft/GPU locally" constraint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vehicle_ai.context_prompt import build_user_prompt
from vehicle_ai.logging_conf import get_logger
from vehicle_ai.normalizer import normalizer_for
from vehicle_ai.schemas import GatewayRequest

from .base import BaseProvider, ProviderError, ProviderResult

logger = get_logger("aurora.gateway.peft")

DEFAULT_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
_SYSTEM = (
    "You are AURORA, an in-vehicle assistant grounded in the ORION-X. "
    "Answer only from the provided telemetry. If data is missing, refuse honestly."
)


class PEFTProvider(BaseProvider):
    """Serve a PEFT LoRA (or the raw base) off the training prompt format."""

    name = "peft"

    def __init__(
        self,
        adapter_dir: str | Path | None,
        *,
        base_model: str = DEFAULT_BASE,
        use_lora: bool = True,
        max_new_tokens: int = 64,
        device: str = "auto",
    ) -> None:
        self._adapter_dir = str(adapter_dir) if adapter_dir else None
        self._base_model = base_model
        self._use_lora = use_lora
        self._max_new_tokens = max_new_tokens
        self._device = device
        self.models = [base_model]

        # Lazily imported / held after first successful load.
        self._tokenizer: Any = None
        self._model: Any = None
        self._load_error: str | None = None

    # -- configuration -----------------------------------------------------

    def is_configured(self) -> bool:
        if self._load_error:
            return False
        if self._model is not None:
            return True
        if self._use_lora and self._adapter_dir is None:
            return False
        if not self._deps_available():
            return False
        if self._use_lora and not (Path(self._adapter_dir) / "adapter_config.json").exists():
            return False
        return True

    def describe(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "models": self.models,
            "configured": self.is_configured(),
            "adapter_dir": self._adapter_dir,
            "use_lora": self._use_lora,
            "base_model": self._base_model,
        }
        if self._load_error:
            d["load_error"] = self._load_error
        return d

    @staticmethod
    def _deps_available() -> bool:
        try:
            import torch  # noqa: F401, E402
            import transformers  # noqa: F401, E402
        except Exception:
            return False
        return True

    # -- lifecycle ---------------------------------------------------------

    def _load(self) -> None:
        """Import torch/transformers/peft once and build the inference model."""
        if self._model is not None:
            return
        if self._load_error:
            raise ProviderError(f"{self.name}: {self._load_error}")

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - local env has no torch
            self._load_error = f"torch/transformers unavailable: {exc}"
            raise ProviderError(self._load_error) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._base_model)
            model = AutoModelForCausalLM.from_pretrained(self._base_model)
        except Exception as exc:
            self._load_error = f"base model load failed: {exc}"
            raise ProviderError(self._load_error) from exc

        if self._use_lora and self._adapter_dir:
            try:
                from peft import PeftModel

                adapt = Path(self._adapter_dir)
                if not (adapt / "adapter_config.json").exists():
                    raise FileNotFoundError(f"no adapter_config.json in {adapt}")
                model = PeftModel.from_pretrained(model, self._adapter_dir)
            except Exception as exc:
                self._load_error = f"adapter load failed: {exc}"
                raise ProviderError(self._load_error) from exc

        self._tokenizer = tokenizer
        self._model = model
        self._load_error = None
        logger.info(
            "peft provider loaded base=%s lora=%s adapter=%s",
            self._base_model,
            self._use_lora,
            self._adapter_dir,
        )

    # -- inference ---------------------------------------------------------

    def _build_prompt(self, req: GatewayRequest) -> str:
        text = build_user_prompt(self._canonical_context(req), self._question(req))
        return (
            '<|im_start|>system\n'
            f'{_SYSTEM}<|im_end|>\n'
            f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n'
        )

    @staticmethod
    def _question(req: GatewayRequest) -> str:
        for m in reversed(req.messages):
            if m.role == "user":
                return m.content
        return ""

    @staticmethod
    def _canonical_context(req: GatewayRequest):
        """Build the canonical context from enum'd grounding + overrides."""
        g = req.grounding
        telemetry = g.telemetry if g else {}
        source = req.metadata.get("telemetry_source") or "orionx_live"
        charging = bool(req.metadata.get("telemetry_charging"))
        return normalizer_for(source, charging=charging)(telemetry)

    def render_prompt(self, req: GatewayRequest) -> str:
        """The exact prompt this provider would tokenize — no model load needed.

        Lets the v2 harness verify byte-for-byte that `peft` and `peft-base`
        receive the *same* composed prompt (only the adapter differs), without
        needing torch/GPU on the validation machine.
        """
        return self._build_prompt(req)

    async def complete(self, req: GatewayRequest) -> ProviderResult:
        if not self.is_configured():
            raise ProviderError(f"{self.name} not configured")

        if self._model is None:
            self._load()

        start = time.perf_counter()
        prompt = self._build_prompt(req)
        import torch  # noqa: F401

        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        content = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return ProviderResult(
            content=content,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=out.shape[1] - inputs["input_ids"].shape[1],
            latency_ms=latency_ms,
            model=self._base_model,
        )

    async def close(self) -> None:  # pragma: no cover - lifecycle nicety
        return