"""Prometheus metrics primitives shared by all services.

Each service exposes ``/metrics``; an operator can point any Prometheus /
Grafana / Azure Monitor Managed Prometheus at them without custom code.
"""

from __future__ import annotations

import functools
import time
from typing import Callable

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        make_asgi_app,
        start_http_server,
    )
    _AVAILABLE = True
except Exception:  # pragma: no cover - fallback when client missing
    _AVAILABLE = False


def metrics_available() -> bool:
    return _AVAILABLE


def _noop(**_: object) -> Callable:
    return lambda *a, **k: None


class MetricsRegistry:
    """Thin facade so services can import metrics even without the client."""

    def __init__(self) -> None:
        if _AVAILABLE:
            self.requests_total = Counter(
                "aurora_requests_total", "Total public requests", ["service", "endpoint"]
            )
            self.request_latency_ms = Histogram(
                "aurora_request_latency_ms",
                "Request latency",
                ["service", "endpoint"],
                buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
            )
            self.signal_gauge = Gauge(
                "aurora_telemetry_signal",
                "Live vehicle signal",
                ["vehicle_id", "signal"],
            )
            self.drift_gauge = Gauge(
                "aurora_drift_psi", "Telemetry drift PSI", ["vehicle_id", "signal"]
            )
        else:  # pragma: no cover
            self.requests_total = _noop()
            self.request_latency_ms = _noop()
            self.signal_gauge = _noop()
            self.drift_gauge = _noop()

    def observe(self, service: str, endpoint: str) -> Callable:
        """Context-manager-lite wrapper recording latency + request count."""

        def _wrap(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def _inner(*a: object, **k: object) -> object:
                start = time.perf_counter()
                self.requests_total.labels(service, endpoint).inc()
                try:
                    return fn(*a, **k)  # type: ignore[return-value]
                finally:
                    self.request_latency_ms.labels(service, endpoint).observe(
                        (time.perf_counter() - start) * 1000
                    )

            return _inner

        return _wrap


def prometheus_asgi_app() -> object | None:
    if not _AVAILABLE:
        return None
    return make_asgi_app()