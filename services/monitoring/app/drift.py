"""Online statistical drift detection over the telemetry stream.

Two eyes per signal:
* **PSI** (Population Stability Index) — how much the live distribution has
  shifted vs the trained baseline. 0-0.1 no change, 0.1-0.25 small, >0.25
  significant shift → alert.
* **KS two-sample p-value** — a complementary goodness-of-fit signal.

The baseline is either loaded from a committed ``baseline.json`` (computed
from the simulator's normal behaviour) or learned from a warm-up window of
the live stream. Either way the operator sees WHICH signal drifted and by
how much — not just "something changed".
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from vehicle_ai.telemetry_utils import KEY_SIGNALS, ks_pvalue, psi


class DriftDetector:
    def __init__(
        self,
        baseline_frames: int,
        eval_frames: int,
        psi_threshold: float,
        ks_threshold: float,
        baseline_file: str = "",
    ) -> None:
        self.baseline_frames = baseline_frames
        self.eval_frames = eval_frames
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.baseline: dict[str, list[float]] | None = None
        self.baseline_source = "none"
        self.frames: deque[dict] = deque(maxlen=max(baseline_frames, eval_frames) * 2 + 100)
        self._warmup_started = False
        self._latest: dict = {}
        self._load_baseline(baseline_file)

    def _load_baseline(self, baseline_file: str) -> None:
        path = Path(baseline_file)
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            self.baseline = {k: [float(x) for x in v] for k, v in data.get("signals", {}).items()}
            self.baseline_source = "file"

    def ingest(self, frame: dict) -> None:
        self.frames.append(frame)
        sigs = frame.get("signals", {})
        self._latest = sigs
        if self.baseline is None and not self._warmup_started and len(self.frames) >= self.baseline_frames:
            self._warmup_started = True
            self.baseline = {k: [f["signals"][k] for f in self.frames if k in f.get("signals", {})] for k in KEY_SIGNALS}
            self.baseline = {k: v for k, v in self.baseline.items() if v}
            self.baseline_source = "live-warmup"

    def evaluate(self) -> dict:
        if not self.baseline:
            return {
                "status": "warming-up",
                "alerted": False,
                "baseline_source": self.baseline_source,
                "baseline_frames": len(next(iter(self.baseline.values()))) if self.baseline else 0,
            }
        frames = list(self.frames)[-self.eval_frames :]
        observed = {k: [f["signals"][k] for f in frames if k in f.get("signals", {})] for k in self.baseline}
        per_psi: dict[str, float] = {}
        per_ks: dict[str, float] = {}
        for k, base in self.baseline.items():
            obs = observed.get(k, [])
            if len(obs) < 30 or len(base) < 30:
                continue
            per_psi[k] = round(psi(obs, base[-len(obs) :], bins=10), 4)
            per_ks[k] = round(ks_pvalue(obs, base[-len(obs) :]), 5)
        alert_signals = {
            k: per_psi[k]
            for k in per_psi
            if per_psi[k] >= self.psi_threshold or (k in per_ks and per_ks[k] <= self.ks_threshold)
        }
        return {
            "status": "ok" if not alert_signals else "drift-detected",
            "alerted": bool(alert_signals),
            "baseline_source": self.baseline_source,
            "baseline_frames": len(next(iter(self.baseline.values()))),
            "evaluated_frames": len(observed),
            "overall_psi": round(max(per_psi.values()) if per_psi else 0.0, 4),
            "alert_signals": alert_signals,
            "per_signal_psi": per_psi,
            "per_signal_ks_p": per_ks,
        }