"""Helpers for working with telemetry signals across services."""

from __future__ import annotations

import math
import statistics

# Signals we routinely compute drift / stats on.
KEY_SIGNALS = [
    "vehicle_speed_kph",
    "battery_soc_pct",
    "battery_temp_c",
    "coolant_temp_c",
    "motor_power_kw",
    "distance_to_empty_km",
    "tire_pressure_fl_kpa",
    "tire_pressure_fr_kpa",
    "tire_pressure_rl_kpa",
    "tire_pressure_rr_kpa",
    "pack_current_a",
    "battery_voltage_v",
    "odometer_km",
]


def sliding_mean(values: list[float], window: int = 12) -> list[float]:
    """Simple trailing moving average used for spreadsheets + demos."""
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(statistics.fmean(values[lo : i + 1]))
    return out


def psi(observed: list[float], expected: list[float], bins: int = 10) -> float:
    """Population Stability Index: how much did a distribution shift?"""
    if not observed or not expected:
        return 0.0
    lo = min(min(observed), min(expected))
    hi = max(max(observed), max(expected))
    if hi - lo < 1e-9:
        return 0.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    edges[0], edges[-1] = -math.inf, math.inf

    def _hist(data: list[float]) -> list[float]:
        counts = [0.0] * bins
        for v in data:
            for i in range(bins):
                if edges[i] <= v < edges[i + 1] or (i == bins - 1 and edges[i] <= v <= edges[-1]):
                    counts[i] += 1
                    break
        total = sum(counts) or 1.0
        return [max(c / total, 1e-6) for c in counts]

    o, e = _hist(observed), _hist(expected)
    return float(sum((o[i] - e[i]) * math.log(o[i] / e[i]) for i in range(bins)))


def ks_pvalue(observed: list[float], expected: list[float]) -> float:
    """Two-sample two-sided Kolmogorov–Smirnov approximate p-value.

    Uses the asymptotic KS limit (valid for large n) — a pragmatic stand-in
    for scipy that keeps the runtime light.
    """
    a, b = sorted(observed), sorted(expected)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    i = j = 0
    d = 0.0
    while i < n and j < m:
        if a[i] < b[j]:
            d = max(d, (i + 1) / n - j / m)
            i += 1
        else:
            d = max(d, (j + 1) / m - i / n)
            j += 1
    d = max(d, (n - j) / n - m / m if i == n else (m - i) / m - n / n)
    d = abs(d)
    ne = (n * m) / (n + m)
    if ne * d * d > 1e-12:
        lam = d * math.sqrt(max(ne, 1.0))
        p = 2 * math.exp(-2 * lam * lam)
    else:
        p = 1.0
    return min(max(p, 0.0), 1.0)


def aggregate_window(frames: list[dict], window_seconds: float) -> dict:
    """Derive mean/min/max/std per signal across a rolling window."""
    import time

    if not frames:
        return {}
    keys = sorted({k for f in frames for k in f.get("signals", {})})
    mean, mini, maxi, std = {}, {}, {}, {}
    for k in keys:
        vals = [f["signals"][k] for f in frames if k in f.get("signals", {}) and isinstance(f["signals"][k], (int, float)) and math.isfinite(f["signals"][k])]
        if not vals:
            continue
        mean[k] = statistics.fmean(vals)
        mini[k] = min(vals)
        maxi[k] = max(vals)
        std[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "frames": len(frames),
        "window_seconds": round(window_seconds, 2),
        "start_ts": frames[0]["timestamp"],
        "end_ts": frames[-1]["timestamp"],
        "mean": mean,
        "min": mini,
        "max": maxi,
        "std": std,
    }