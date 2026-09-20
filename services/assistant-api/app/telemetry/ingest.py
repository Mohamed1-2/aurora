"""Live telemetry ingestion for the assistant.

The assistant keeps its own rolling window of *real* vehicle frames fed over
WebSocket from the telemetry simulator (10 Hz). This mirrors the production
pattern where the vehicle pushes to the assistant instead of the assistant
polling: answers about "right now" are actually about right now.

If the simulator is unreachable and a replay snapshot is configured (the eval
harness does this for determinism), it falls back to the snapshot.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.telemetry_utils import aggregate_window, KEY_SIGNALS

logger = get_logger("aurora.assistant.telemetry")


def parse_frame(raw: str) -> dict | None:
    """Reject invalid timestamps; omit invalid signals without losing good readings."""
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        return value
    frame=finite(json.loads(raw))
    if not isinstance(frame,dict) or not isinstance(frame.get('signals'),dict):return None
    ts=frame.get('timestamp')
    if not isinstance(ts,(int,float)) or not math.isfinite(ts):return None
    frame['signals']={k:v for k,v in frame['signals'].items() if isinstance(v,(int,float)) and math.isfinite(v)}
    return frame


class TelemetryIngest:
    def __init__(self, ws_url: str, replay_path: str, window_seconds: float = 10.0, max_frames: int = 900) -> None:
        self.ws_url = ws_url
        self.replay_path = replay_path
        self.window_seconds = window_seconds
        self.frames: deque[dict] = deque(maxlen=max_frames)
        self.source: str = "none"
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._listener_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listener_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._listen_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self.replay_path and self.source != "replay":
                    self._load_replay()
                    logger.warning("telemetry WS unavailable, using replay snapshot", extra={"payload": {"error": str(exc)}})
                await asyncio.sleep(5.0)

    async def _listen_once(self) -> None:
        import websockets

        async with websockets.connect(self.ws_url, ping_interval=20, max_size=2**20) as ws:
            self.source = "live"
            logger.info("telemetry stream connected", extra={"payload": {"url": self.ws_url}})
            async for raw in ws:
                try:
                    frame = parse_frame(raw)
                except json.JSONDecodeError:
                    continue
                if frame is not None:
                    self.frames.append(frame)
                self._stop.clear()

    def _load_replay(self) -> None:
        path = Path(self.replay_path)
        if not path.exists():
            return
        with path.open() as f:
            data = json.load(f)
        if "signals" in data:
            data["seq"] = data.get("seq", 0)
            self.frames.append(data)
        self.source = "replay"

    # -- grounding ingestion -------------------------------------------------

    def grounding(self) -> dict[str, Any]:
        if not self.frames:
            return {}
        frames = list(self.frames)
        latest = frames[-1]
        window_seconds = min(self.window_seconds, len(frames) * 0.1)
        agg = aggregate_window(frames[-int(window_seconds / 0.1) :], window_seconds)
        now = latest.get("signals", {})
        prev = frames[-30].get("signals", {}) if len(frames) >= 30 else {}
        trending: dict[str, float] = {}
        for k in KEY_SIGNALS:
            if k in now and k in prev and isinstance(now[k], (int, float)) and isinstance(prev[k], (int, float)):
                delta = now[k] - prev[k]
                trending[k] = round(delta * (60.0 / 3.0), 2)  # per-minute slope over ~3s

        dtcs = [
            {"code": d.get("code"), "description": d.get("description", ""), "severity": d.get("severity", "warning")}
            for d in latest.get("active_dtcs", [])
        ]
        return {
            "vehicle_id": latest["vehicle_id"],
            "timestamp": latest["timestamp"],
            "mode": latest.get("driving_mode", "normal"),
            "charging": latest.get("charging", False),
            "signals": now,
            "mean": agg.get("mean", {}),
            "min": agg.get("min", {}),
            "max": agg.get("max", {}),
            "trending": trending,
            "dtcs": dtcs,
            "warnings": latest.get("warnings", []),
            "source": self.source,
        }

    def live_summary(self) -> dict[str, Any]:
        """Compact payload for the dashboard's live cards."""
        g = self.grounding()
        if not g:
            return {"status": "no-telemetry", "vehicle_id": "?"}
        want = [
            "battery_soc_pct", "distance_to_empty_km", "vehicle_speed_kph",
            "battery_temp_c", "coolant_temp_c", "motor_power_kw",
            "tire_pressure_fl_kpa", "tire_pressure_rr_kpa", "charge_rate_kw",
        ]
        return {
            "vehicle_id": g["vehicle_id"],
            "mode": g["mode"],
            "charging": g["charging"],
            "signals": {k: g["signals"].get(k) for k in want},
            "dtc_codes": [d["code"] for d in g["dtcs"]],
            "warnings": [w for w in g["warnings"]],
            "source": g["source"],
            "window_seconds": self.window_seconds,
        }

    def signal_series(self, signal: str, max_age_s: float = 45.0, stride_s: float = 1.0) -> list[tuple[float, float]]:
        """(timestamp, value) samples of one signal across the cached buffer.

        Signals are decimated to ~one sample per ``stride_s`` (frames arrive at
        10 Hz) so slope/window detectors get a stable, aligned series. Only
        frames that actually carry the signal are returned — a signal that is
        simply absent yields ``[]`` (callers must treat empty as "no data",
        never as zero).
        """
        step = max(1, int(round(stride_s * 10)))
        out: list[tuple[float, float]] = []
        frames = list(self.frames)
        now = time.time()
        for i in range(len(frames) - 1, -1, -step):
            f = frames[i]
            v = f.get("signals", {}).get(signal)
            ts = f.get("timestamp")
            if ts is None or now - ts > max_age_s:
                break
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN
                continue
            out.append((ts, v))
        out.reverse()
        return out

    def has_signal(self, signal: str) -> bool:
        """True if the most recent frame carries the given signal."""
        if not self.frames:
            return False
        v = self.frames[-1].get("signals", {}).get(signal)
        return v is not None