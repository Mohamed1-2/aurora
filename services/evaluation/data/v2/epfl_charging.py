"""AURORA v2 — EPFL Level-3 EV charging ingestor (workbook → aligned sessions).

Reads the EPFL DESL Level-3 EV Charging dataset (MIT licensed) from two Excel
sheets:

    Measurement_data.xlsx  -> 1-minute, timestamp-aligned charging measurements
    Session_data.xlsx      -> session-level summaries (all fields complete)

and emits normalized ``ChargingSession`` records. Each session keeps only rows
that actually carry data (the workbook is ~94 % empty placeholders) and only
within a **single real charging session** — never combining unrelated sessions.

Coherence: ``timestamp_aligned`` / ``state_reconstruction=measured_session``.
"""

from __future__ import annotations

from pathlib import Path

try:
    import pandas as pd  # noqa: E402
except Exception as _exc:  # pragma: no cover
    pd = None  # type: ignore
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

from .base import ChargingSession, Provenance

_MEASURE_FILE = "Measurement_data.xlsx"
_SESSION_FILE = "Session_data.xlsx"

# Column name in measurement sheet -> canonical name + scale (W->kW, Wh->kWh)
_MEAS_MAP = {
    "P (W)": ("charging_power_kw", 1e-3),
    "Pset (W)": ("power_setpoint_kw", 1e-3),
    "Preq (W)": ("requested_power_kw", 1e-3),
    "TotalCapacity": ("total_capacity_wh", 1.0),
    "RemainingCapacity": ("remaining_capacity_wh", 1.0),
    "BulkCapacity": ("bulk_capacity_wh", 1.0),
    "Energy (Wh)": ("energy_charged_wh", 1.0),
    "SOC (%)": ("soc_pct", 1.0),
    "Ctrl_t (0 = False, 1 = True)": ("ctrl_t", 1.0),
    "Ctrl_s (0 = False, 1 = True)": ("ctrl_s", 1.0),
}


def require_pandas() -> None:  # pragma: no cover
    if pd is None:
        raise ImportError(
            "pandas and openpyxl are required to read the EPFL workbook. "
            "Install with: pip install pandas openpyxl"
        ) from _IMPORT_ERROR


class EPFLChargingIngestor:
    """Normalize EPFL charging data into aligned sessions."""

    def __init__(self, dataset_dir: Path | str, max_sessions: int | None = None) -> None:
        require_pandas()
        self.dir = Path(dataset_dir)
        self.max_sessions = max_sessions

    def _measurement_path(self) -> Path:
        return self.dir / _MEASURE_FILE

    def _session_path(self) -> Path:
        return self.dir / _SESSION_FILE

    def ingest(self) -> list[ChargingSession]:
        meas = pd.read_excel(self._measurement_path())
        sess = pd.read_excel(self._session_path())

        # Drop empty placeholder rows (no SOC/power data).
        meas = meas.dropna(subset=["SOC (%)"])
        sessions: list[ChargingSession] = []
        for sid, group in meas.groupby("Session"):
            if self.max_sessions is not None and len(sessions) >= self.max_sessions:
                break
            rows: list[dict[str, float]] = []
            for _, r in group.iterrows():
                row: dict[str, float] = {}
                ok = True
                for src_col, (canon_name, scale) in _MEAS_MAP.items():
                    v = r.get(src_col)
                    if v is None or (isinstance(v, float) and v != v):  # NaN
                        ok = False
                        break
                    row[canon_name] = round(float(v) * scale, 4)
                if not ok:
                    continue
                # timestamp (minute-level)
                row["minute_offset_s"] = float(r.get("Date and time", 0).timestamp()) if hasattr(r.get("Date and time", 0), "timestamp") else 0.0
                rows.append(row)
            if not rows:
                continue
            # session summary from the session sheet
            sm_ref = sess[sess["Session"] == sid]
            summary: dict[str, float] = {
                "soc_arrival_pct": float(sm_ref["SOC arrival"].iloc[0]),
                "soc_departure_pct": float(sm_ref["SOC departure"].iloc[0]),
                "stay_min": float(sm_ref["Stay (min)"].iloc[0]),
                "energy_charged_kwh": float(sm_ref["Energy (Wh)"].iloc[0]) / 1000.0,
                "max_power_kw": float(sm_ref["Pmax (W)"].iloc[0]) / 1000.0,
                "max_requested_power_kw": float(sm_ref["Preq_max (W)"].iloc[0]) / 1000.0,
                "total_capacity_kwh": float(sm_ref["TotalCapacity"].iloc[0]) / 1000.0,
                "bulk_capacity_kwh": float(sm_ref["BulkCapacity"].iloc[0]) / 1000.0,
                "controlled_session": float(sm_ref["Controlled session (0=False, 1=True)"].iloc[0]),
            }
            prov = Provenance(
                source_dataset="EPFL_L3_CHARGING",
                source_record=f"{_MEASURE_FILE} / {_SESSION_FILE}",
                state="charging",
                vehicle_id=None,
                signals_used=sorted({"charging_power_kw", "power_setpoint_kw",
                                     "requested_power_kw", "soc_pct",
                                     "energy_charged_wh", "total_capacity_wh",
                                     "remaining_capacity_wh", "bulk_capacity_wh",
                                     "ctrl_t", "ctrl_s"}),
                coherence="timestamp_aligned",
                state_reconstruction="measured_session",
                session_id=str(sid),
            )
            sessions.append(ChargingSession(
                session_id=str(sid),
                measurements=rows,
                summary=summary,
                provenance=prov,
            ))
        return sessions
