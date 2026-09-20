"""Telemetry utilities: windowing/aggregation and statistical drift math."""

import math

import pytest

from vehicle_ai.telemetry_utils import aggregate_window, psi, sliding_mean


def _frames(n: int, base: float, step: float = 1.0):
    out = []
    for i in range(n):
        out.append({"timestamp": i, "signals": {"battery_soc_pct": base + step * i, "vehicle_speed_kph": 50.0}})
    return out


def test_sliding_mean():
    vals = [1, 2, 3, 4, 5, 6]
    sm = sliding_mean(vals, window=3)
    assert len(sm) == len(vals)
    assert sm[-1] == pytest.approx(5, abs=1e-9)
    assert sm[0] == pytest.approx(1, abs=1e-9)


def test_aggregate_window():
    w = aggregate_window(_frames(10, 60.0, 1.0), window_seconds=1.0)
    assert w["frames"] == 10
    assert w["mean"]["battery_soc_pct"] == pytest.approx(64.5, abs=1e-9)
    assert w["min"]["battery_soc_pct"] == 60.0
    assert w["max"]["battery_soc_pct"] == 69.0
    assert w["std"]["vehicle_speed_kph"] == 0.0


def test_aggregate_empty():
    assert aggregate_window([], 1.0) == {}


def test_psi_identical_distributions():
    assert psi([1, 2, 3, 4, 5] * 20, [1, 2, 3, 4, 5] * 20, bins=10) == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_shift():
    p = psi([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 50, [60, 70, 80, 90, 100, 110, 120, 130, 140, 150] * 50, bins=10)
    assert p > 5.0
    assert math.isfinite(p)