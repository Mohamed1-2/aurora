"""Generate the committed drift-detection baseline from normal driving.

Runs the simulator's own physics model for ~10 virtual minutes and exports
per-signal value lists. The monitoring service loads this at startup so
drift detection is meaningful from the very first frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services/telemetry-simulator"))

from app.engine.vehicle_model import VehicleModel  # noqa: E402
from vehicle_ai.telemetry_utils import KEY_SIGNALS  # noqa: E402

model = VehicleModel("ORION-X-TEST-01", seed=42)
dt = 0.1
minutes = 10
per_signal: dict[str, list[float]] = {k: [] for k in KEY_SIGNALS}

model._session_wh = 0.0
model._session_km = 0.0

for i in range(int(minutes * 60 / dt)):
    frame = model.tick(dt)
    for k in KEY_SIGNALS:
        v = frame["signals"].get(k)
        if isinstance(v, (int, float)) and v == v:
            per_signal[k].append(v)
    # subsample every frame (10 Hz) -> plenty of points

out = {
    "vehicle_id": "ORION-X-TEST-01",
    "generated_at": 0,
    "note": "normal driving baseline from vehicle physics model (seed 42)",
    "signals": {k: [round(x, 2) for x in v] for k, v in per_signal.items()},
}

dest = Path(__file__).resolve().parent.parent / "services/monitoring/baseline.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out))
print(f"wrote {dest} with points: { {k: len(v) for k, v in per_signal.items()} }")