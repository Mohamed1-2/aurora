"""Group aligned 10 s TUM bins into contiguous 300 s driving windows.

Each emitted ``RawWindow`` holds the (up to 30) constituent bins so downstream
feature extraction can aggregate from the same per-bin signals every time. The
bins came from ``TUMDrivingIngestor`` gated on the dense 5-core signal set:
``{soc_pct, speed_kph, hv_aux_power_w, outside_temp_c, coolant_temp_c}``.

Definitions:
* run         — maximal sequence of bins from one vehicle with ≤ ``max_gap_s``
                between consecutive bin starts (a single continuous drive).
* window      — the first ``W/bin_s`` consecutive bins of a run (non-overlapping,
                tail shorter than the window is dropped).
* driving     — window kept iff moving_fraction ≥ threshold and the window is a
                net discharge (observed soc_drop_pct > 0, ≤ sane ceiling).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

SOC_QUANT_PCT = 0.4  # verified hv_soc quantization step (all TUM vehicles)


@dataclass(frozen=True)
class WindowConfig:
    window_s: float = 300.0
    bin_s: float = 10.0
    max_gap_s: float = 30.0
    min_moving_fraction: float = 0.8
    min_soc_drop_pct: float = 0.0
    max_soc_drop_pct: float = 15.0


def _iso_to_dt(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def group_runs(windows, max_gap_s: float) -> list[list]:
    runs: list[list] = []
    cur: list = []
    prev: dt.datetime | None = None
    for w in sorted(windows, key=lambda x: _iso_to_dt(x.window_start_utc)):
        start = _iso_to_dt(w.window_start_utc)
        if prev is not None and (start - prev).total_seconds() > max_gap_s:
            runs.append(cur)
            cur = []
        cur.append(w)
        prev = start
    if cur:
        runs.append(cur)
    return runs


@dataclass
class RawWindow:
    vehicle_id: str
    episode_id: str
    run_id: str
    split: str = ""
    window_dur_s: float = 0.0
    moving_fraction: float = 0.0
    soc_start_pct: float = 0.0
    soc_end_pct: float = 0.0
    start_utc: str = ""
    end_utc: str = ""
    n_bins: int = 0
    observed_soc_drop_pct: float = 0.0
    quant_steps: float = 0.0
    bins: list[dict] = field(default_factory=list)


def _window_from_run(run: list, cfg: WindowConfig, run_index: int) -> list[RawWindow]:
    """Slice non-overlapping training windows out of one contiguous run."""
    out: list[RawWindow] = []
    n_bins_per = max(1, int(round(cfg.window_s / cfg.bin_s)))
    for k in range(0, len(run) - n_bins_per + 1, n_bins_per):
        chunk = run[k : k + n_bins_per]
        if len(chunk) < n_bins_per:
            continue
        speeds = [b.signals["speed_kph"] for b in chunk]
        moving = sum(1 for v in speeds if v > 0) / len(chunk)
        soc_start = chunk[0].signals["soc_pct"]
        soc_end = chunk[-1].signals["soc_pct"]
        drop = soc_start - soc_end
        if moving < cfg.min_moving_fraction:
            continue
        if not (cfg.min_soc_drop_pct < drop <= cfg.max_soc_drop_pct):
            continue
        first = _iso_to_dt(chunk[0].window_start_utc)
        last = _iso_to_dt(chunk[-1].window_start_utc)
        end = last + dt.timedelta(seconds=cfg.bin_s)
        out.append(RawWindow(
            vehicle_id=chunk[0].vehicle_id,
            episode_id=chunk[0].episode_id,
            run_id=f"{chunk[0].vehicle_id}-{first.date().isoformat()}-r{run_index:03d}",
            window_dur_s=len(chunk) * cfg.bin_s,
            moving_fraction=round(moving, 4),
            soc_start_pct=soc_start,
            soc_end_pct=soc_end,
            start_utc=chunk[0].window_start_utc,
            end_utc=end.isoformat(),
            n_bins=len(chunk),
            observed_soc_drop_pct=round(drop, 4),
            quant_steps=round(drop / SOC_QUANT_PCT, 2),
            bins=[dict(b.signals) for b in chunk],
        ))
    return out


def build_windows(windows, cfg: WindowConfig | None = None) -> list[RawWindow]:
    cfg = cfg or WindowConfig()
    runs = group_runs(windows, cfg.max_gap_s)
    out: list[RawWindow] = []
    for run_index, run in enumerate(runs):
        out.extend(_window_from_run(run, cfg, run_index))
    return out