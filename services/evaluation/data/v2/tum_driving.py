"""AURORA v2 — TUM driving-telemetry ingestor (raw parquet → aligned windows).

Reads the TUM *Electric Vehicle Usage Data Set for Battery Aging Studies*
(Schreiber et al., TUM, CC-BY-NC-4.0) raw UDS parquet logs:

    data/uds_data/{vehicle}.parquet   (long format: vehicle_id, time, value_id, value)

and resamples the per-signal time series onto a common wall-clock window grid so
that every emitted ``DrivingWindow`` holds **temporally-aligned, cross-signal
simultaneous** values measured from the same vehicle in the same window — the
exact "don't splice SOC/speed/temp from different moments" integrity property
required for v2 (docs/dataset-v2-inspection-report.md §2.7).

Coherence: ``timestamp_aligned`` / ``state_reconstruction=temporal_binned_window``.
Corrupted timestamps (year 2087 outliers) and out-of-2022/23 spans are dropped.
Only signals present for that vehicle are used.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as _exc:  # pragma: no cover - import guard
    pa = None  # type: ignore
    pq = None  # type: ignore
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

from .base import DrivingWindow, Provenance

# value_id -> canonical AURORA context name (measured only).
SIGNAL_MAP: dict[int, str] = {
    4: "speed_kph",          # vehicle_speed [km/h]
    15: "outside_temp_c",    # ambient_air_temp [°C]
    56: "hv_aux_power_w",    # hv_aux_power [W]
    900: "soc_pct",          # hv_soc [%]
    961: "motor_stator_temp_c",   # temp_rear_motor_stator [°C]
    1200: "pack_voltage_v",  # hv_battery_voltage [V]
    1208: "battery_temp_min_c",   # hv_temp_min [°C]
    1209: "battery_temp_c",  # hv_temp_max [°C]
    1265: "motor_rotor_temp_c",   # temp_rotor_rear_motor [°C]
    1269: "coolant_temp_c",  # coolant_temp_inverter_inlet [°C]
    1272: "battery_temp_inlet_c", # hv_battery_temp_inlet [°C]
    1273: "battery_temp_outlet_c",# hv_battery_temp_outlet [°C]
}

# Canonical name -> source value_id (reverse map).
_NAME_TO_ID: dict[str, int] = {v: k for k, v in SIGNAL_MAP.items()}

_WINDOW_S = 10.0  # resample grid (seconds)
_MIN_YEAR = 2022
_MAX_YEAR = 2023

# Default emission gate: windows must contain all of these signals to be
# considered coherent cross-signal 10 s states (LoRA answer-grounding set).
DEFAULT_CORE_SIGNALS = frozenset(
    {"soc_pct", "speed_kph", "battery_temp_c", "coolant_temp_c", "outside_temp_c"}
)

DRIVING_SIGNAL_DOC = {
    # canonical name -> (unit, human description) for answer grounding
    "speed_kph": ("km/h", "vehicle speed"),
    "outside_temp_c": ("°C", "outside air temperature"),
    "soc_pct": ("%", "state of charge"),
    "battery_temp_c": ("°C", "battery maximum temperature"),
    "battery_temp_min_c": ("°C", "battery minimum temperature"),
    "coolant_temp_c": ("°C", "coolant temperature at inverter inlet"),
    "pack_voltage_v": ("V", "high-voltage battery pack voltage"),
    "hv_aux_power_w": ("W", "high-voltage auxiliary power draw"),
    "motor_stator_temp_c": ("°C", "rear motor stator temperature"),
    "motor_rotor_temp_c": ("°C", "rear motor rotor temperature"),
    "battery_temp_inlet_c": ("°C", "battery pack coolant inlet temperature"),
    "battery_temp_outlet_c": ("°C", "battery pack coolant outlet temperature"),
}


def require_pyarrow() -> None:  # pragma: no cover
    if pq is None:
        raise ImportError(
            "pyarrow is required to read the TUM parquet logs. "
            "Install it with: pip install pyarrow"
        ) from _IMPORT_ERROR


def _ns_to_epoch_utc(ns: int) -> _dt.datetime:
    """Convert an int nanosecond timestamp to an aware UTC datetime."""
    return _dt.datetime.fromtimestamp(ns / 1e9, tz=_dt.timezone.utc)


class TUMDrivingIngestor:
    """Resample TUM UDS parquet logs into aligned 10 s driving windows."""

    def __init__(
        self,
        uds_dir: Path | str,
        window_s: float = _WINDOW_S,
        min_year: int = _MIN_YEAR,
        max_year: int = _MAX_YEAR,
        core_signals: set[str] | None = None,
    ) -> None:
        require_pyarrow()
        self.uds_dir = Path(uds_dir)
        self.window_s = window_s
        self.min_year = min_year
        self.max_year = max_year
        # ``core_signals`` selects which signals a 10 s aligned window must
        # contain to be emitted. Default keeps the original (battery-temp-gated)
        # set for back-compat; callers that need more data pass the dense core.
        self.core_signals = frozenset(core_signals) if core_signals is not None else DEFAULT_CORE_SIGNALS

    def _read_vehicle(self, vehicle: str) -> list[tuple[int, int, int, float]]:
        """Load one vehicle parquet -> list of (epoch_ns, value_id, value)."""
        pqf = pq.ParquetFile(str(self.uds_dir / f"{vehicle}.parquet"))
        t = pqf.read(columns=["time"]).column("time").to_numpy()
        vid = pqf.read(columns=["value_id"]).column("value_id").to_pylist()
        val = pqf.read(columns=["value"]).column("value").to_numpy()
        epoch_ns = t.astype("datetime64[ns]").astype("int64")
        return list(zip(epoch_ns.tolist(), vid, val.tolist()))

    def _bin_vehicle(self, vehicle: str, rows: list[tuple[int, int, int, float]]) -> list[DrivingWindow]:
        """Bin rows into 10 s windows; each window holds nearest-sample per signal."""
        # Filter to sane years and to known signals.
        lo = _dt.datetime(self.min_year, 1, 1, tzinfo=_dt.timezone.utc)
        hi = _dt.datetime(self.max_year, 12, 31, 23, 59, 59, tzinfo=_dt.timezone.utc)
        lo_ns = int(lo.timestamp() * 1e9)
        hi_ns = int(hi.timestamp() * 1e9)
        win_ns = int(self.window_s * 1e9)

        # group rows by window bucket
        buckets: dict[int, dict[int, tuple[int, float]]] = {}  # bucket -> {value_id: (epoch_ns, value)}
        for epoch_ns, v_id, value in rows:
            if v_id not in SIGNAL_MAP:
                continue
            if epoch_ns < lo_ns or epoch_ns > hi_ns:
                continue
            bucket = epoch_ns // win_ns
            d = buckets.get(bucket)
            if d is None:
                d = {}
                buckets[bucket] = d
            # keep nearest sample to window start (±) — since all rows fall in
            # this window, just prefer the one closest to window centre.
            cur = d.get(v_id)
            if cur is None:
                d[v_id] = (epoch_ns, value)
            else:
                # keep the sample nearest to window centre
                centre = bucket * win_ns + win_ns // 2
                if abs(epoch_ns - centre) < abs(cur[0] - centre):
                    d[v_id] = (epoch_ns, value)

        # emit windows in time order, building episodes (contiguous windows)
        windows: list[DrivingWindow] = []
        for bucket in sorted(buckets):
            d = buckets[bucket]
            start_ns = bucket * win_ns
            start_utc = _ns_to_epoch_utc(start_ns).isoformat()
            signals = {SIGNAL_MAP[vid]: round(v, 4) for vid, (_t, v) in d.items()}
            # Only keep windows with the core signals we ground answers on.
            core = self.core_signals
            if not core.issubset(signals):
                continue
            # episode id = vehicle + calendar date (UTC) of window
            day = _ns_to_epoch_utc(start_ns).date().isoformat()
            episode_id = f"{vehicle}-{day}"
            prov = Provenance(
                source_dataset="TUM_UDS",
                source_record=f"data/uds_data/{vehicle}.parquet",
                state="driving",
                vehicle_id=vehicle,
                signals_used=sorted(signals),
                coherence="timestamp_aligned",
                state_reconstruction="temporal_binned_window",
                window_start_utc=start_utc,
                window_duration_s=self.window_s,
                episode_id=episode_id,
            )
            windows.append(DrivingWindow(
                vehicle_id=vehicle,
                window_start_utc=start_utc,
                window_duration_s=self.window_s,
                episode_id=episode_id,
                signals=signals,
                provenance=prov,
            ))
        return windows

    def vehicles(self) -> list[str]:
        return sorted(p.stem for p in self.uds_dir.glob("*.parquet"))

    def ingest(self, vehicles: list[str] | None = None) -> list[DrivingWindow]:
        out: list[DrivingWindow] = []
        for v in vehicles or self.vehicles():
            rows = self._read_vehicle(v)
            out.extend(self._bin_vehicle(v, rows))
        return out
