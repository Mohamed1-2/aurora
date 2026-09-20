"""Path bootstrap + cross-service module loading for the calibration package.

services/<name> packages are not importable from the repo root (services/ has no
__init__.py), mirroring how tum_replay loads the TUM ingestor. simparams is
loaded by explicit file spec under a unique module name so it can never collide
with another ``app.engine.simparams`` in the same interpreter.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "services" / "evaluation"
_SIM_DIR = _REPO_ROOT / "services" / "telemetry-simulator"


def repo_root() -> Path:
    return _REPO_ROOT


def add_evaluation_to_path() -> None:
    p = str(_EVAL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_tum_driving():
    add_evaluation_to_path()
    return importlib.import_module("data.v2.tum_driving")


def load_simparams():
    engine = load_ws1_engine()
    return engine


def load_ws1_engine():
    """Load WS1 app.engine (simparams + energy_engine) under a synthetic package.

    Registered in sys.modules as ``aurora_ws1.<module>`` so energy_engine's
    ``from .simparams import ...`` resolves without colliding with any other
    interpreter-side ``app`` package.
    """
    import types

    pkg = types.ModuleType("aurora_ws1")
    pkg.__path__ = [str(_SIM_DIR / "app" / "engine")]
    sys.modules["aurora_ws1"] = pkg
    for name in ("simparams", "energy_engine"):
        p = _SIM_DIR / "app" / "engine" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"aurora_ws1.{name}", p)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError(f"cannot load aurora_ws1.{name} from {p}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["aurora_ws1.energy_engine"]