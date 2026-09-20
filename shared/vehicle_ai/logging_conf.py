"""Structured JSON logging + metrics conventions.

Every service logs structured events through ``get_logger`` so that Fluentd /
Loki / Kusto can ingest them uniformly in production.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """A minimal, fast JSON log formatter (no external deps)."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, "service", None),
            "trace_id": getattr(record, "trace_id", None),
            "vehicle_id": getattr(record, "vehicle_id", None),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            entry.update(extra)
        return json.dumps(entry, default=str)


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Tell uvicorn to reuse our JSON handler too, keeps the tail clean.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False


def get_logger(name: str, service: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.service = service  # type: ignore[attr-defined]
    return logger