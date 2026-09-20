"""MLflow integration: tracking + model registry.

Every evaluation is logged as a run (params/metrics/artifacts) and, on
acceptance, a new version of the "assistant skill pack" is registered —
tying a *versioned prompt + RAG index + evaluation score* to a deployable
artifact. That is the Model Registry loop the architecture diagram promises.

Fully optional: if MLflow is unreachable (or not installed) the harness
still produces a complete local report and the run continues.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _mlflow():
    try:
        import mlflow  # type: ignore
        return mlflow
    except Exception:
        return None


def log_run(
    tracking_uri: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: dict[str, str],
    register: bool = False,
) -> tuple[str | None, str | None]:
    """Log an eval run; returns (mlflow_run_id, registered_model_name|None)."""
    mlflow = _mlflow()
    if mlflow is None:
        return None, None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("aurora-assistant-evals")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics(metrics)
            for name, path in artifact_paths.items():
                p = Path(path)
                if p.is_file():
                    mlflow.log_artifact(str(p), artifact_path=f"artifacts/{name}")
            model_name = None
            if register and artifact_paths.get("package"):
                model_name = "aurora-assistant-skillpack"
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/artifacts/package",
                    name=model_name,
                    tags={"prompt_version": params.get("prompt_version", ""), "pass_rate": str(metrics.get("pass_rate", 0.0))},
                )
            return run_id, model_name
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped (unreachable): {exc}")
        return None, None


def package_config(assistant_env: dict[str, Any]) -> bytes:
    """The 'model' that gets registered: prompt + rag index + routing config."""
    return json.dumps(assistant_env, indent=2).encode()