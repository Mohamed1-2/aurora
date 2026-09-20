"""PEFT (LoRA) gateway provider tests — must pass with no torch/GPU installed."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "llm-gateway"))

from vehicle_ai.schemas import GatewayRequest, Grounding, Message  # noqa: E402

from app.providers.peft import PEFTProvider  # noqa: E402
from app.providers.base import ProviderError  # noqa: E402

ADAPTER = "%s"


def _req(**kw) -> GatewayRequest:
    msgs = kw.pop("messages", [Message(role="user", content="Summarize my charging session.")])
    grounding = kw.pop("grounding", None)
    return GatewayRequest(messages=msgs, grounding=grounding, **kw)


def _req_with_charging_ctx(source: str = "EPFL_L3_CHARGING") -> GatewayRequest:
    g = Grounding(
        telemetry={
            "soc_arrival_pct": 12.0,
            "soc_departure_pct": 91.0,
            "stay_min": 36.0,
            "energy_charged_kwh": 39.254,
        }
    )
    return GatewayRequest(
        messages=[Message(role="user", content="Summarize my charging session.")],
        grounding=g,
        metadata={"telemetry_source": source, "telemetry_charging": True},
    )


class TestNoInferenceRequired:
    """The provider must degrade cleanly when torch/transformers are absent.

    Runs in a subprocess with ``torch``/``transformers``/``peft`` import-blocked
    so the assertions hold regardless of what is installed in the venv.
    """

    _CODE = """
import sys
sys.path.insert(0, 'services/llm-gateway')
import builtins
_blocked = {'torch', 'transformers', 'peft'}
_real_import = builtins.__import__
def _blocked_import(name, *a, **k):
    if name.split('.')[0] in _blocked:
        raise ImportError(f'blocked {name}')
    return _real_import(name, *a, **k)
builtins.__import__ = _blocked_import

from vehicle_ai.schemas import GatewayRequest, Grounding, Message
from app.providers.peft import PEFTProvider
from app.providers.base import ProviderError
import asyncio

ADAPTER = "artifacts/finetune/v2/adapter"
p = PEFTProvider(ADAPTER)
assert p.is_configured() is False
assert p.describe()['configured'] is False

p2 = PEFTProvider(None, use_lora=False)
assert p2.is_configured() is False

g = Grounding(telemetry={"soc_arrival_pct": 12.0})
req = GatewayRequest(messages=[Message(role='user', content='Summarize my charging session.')], grounding=g, metadata={"telemetry_source": 'EPFL_L3_CHARGING', 'telemetry_charging': True})
try:
    asyncio.run(PEFTProvider(ADAPTER).complete(req))
    raise SystemExit('expected ProviderError')
except ProviderError:
    pass
print('NO_TORCH_DEGRADE_OK')
"""

    def test_degrades_cleanly_without_torch(self):
        import os
        import subprocess
        exe = ROOT / ".venv" / "bin" / "python"
        code = self._CODE.replace(
            'sys.path.insert(0, \'services/llm-gateway\')',
            f'sys.path.insert(0, {str(ROOT / "services/llm-gateway")!r})',
        )
        r = subprocess.run(
            [str(exe), "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        assert "NO_TORCH_DEGRADE_OK" in r.stdout, f"stderr: {r.stderr}"

    def test_models_label(self):
        p = PEFTProvider(ADAPTER)
        assert p.models == ["HuggingFaceTB/SmolLM2-135M-Instruct"]

    def test_prompt_builds_from_charging_ctx(self):
        p = PEFTProvider(ADAPTER)
        req = _req_with_charging_ctx()
        prompt = p._build_prompt(req)
        assert prompt.startswith('<|im_start|>system\nYou are AURORA')
        assert '<|im_start|>user\nTelemetry: soc_arrival_pct 12.0; soc_departure_pct 91.0' in prompt
        assert '<|im_start|>assistant\n' in prompt

    def test_prompt_uses_training_template(self):
        p = PEFTProvider(ADAPTER)
        d = GatewayRequest(
            messages=[Message(role="user", content="What's my SOC?")],
            grounding=Grounding(telemetry={"battery_soc_pct": 68.5}),
            metadata={"telemetry_source": "orionx_live", "telemetry_charging": False},
        )
        prompt = p._build_prompt(d)
        assert 'soc_pct 68.5' in prompt
        assert 'battery_soc_pct' not in prompt
        assert 'Question: What\'s my SOC?' in prompt

    def test_unknown_source_raises_loudly(self):
        p = PEFTProvider(ADAPTER)
        d = GatewayRequest(
            messages=[Message(role="user", content="hi")],
            grounding=Grounding(telemetry={"x": 1}),
            metadata={"telemetry_source": "bogus", "telemetry_charging": False},
        )
        with pytest.raises(KeyError):
            p._canonical_context(d)