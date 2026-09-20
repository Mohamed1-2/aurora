"""Deterministic unit tests for the WS2 calibration prep (no raw TUM data needed).

Focus: contracts that must hold BEFORE any ML fitting — capacity/provenance
table, split rules, positive-drop target convention, window builder grouping,
and the physics baseline substitutions (measured aux, hvac=0).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from calibration.vehicle_capacity import (  # noqa: E402
    DEFAULT_CAPACITY_KWH, VEHICLE_CAPACITIES, capacity_for, capacity_table,
)
from calibration.split import SPLIT_RULES, assign_split, build_manifest  # noqa: E402
from calibration.windowing import WindowConfig, build_windows, group_runs  # noqa: E402
from calibration.physics import PhysicsBaseline  # noqa: E402


def _bin(vehicle, start_iso, soc, speed, aux=500.0, ambient=10.0, coolant=20.0, **extra):
    from data.v2.base import DrivingWindow, Provenance  # type: ignore
    signals = {"soc_pct": soc, "speed_kph": speed, "hv_aux_power_w": aux,
               "outside_temp_c": ambient, "coolant_temp_c": coolant}
    signals.update(extra)
    prov = Provenance(source_dataset="TUM_UDS", source_record="x.parquet", state="driving",
                      vehicle_id=vehicle)
    return DrivingWindow(vehicle_id=vehicle, window_start_utc=start_iso, window_duration_s=10.0,
                         episode_id=f"{vehicle}-2022-12-12", signals=signals, provenance=prov)


def _run(vehicle, start, n, step_s=10.0, speed=60.0, soc_drop_per_step=0.0, **extra):
    from datetime import datetime, timedelta, timezone
    t = datetime.fromisoformat(start)
    out = []
    for i in range(n):
        soc = soc_drop_per_step * (n - i)
        out.append(_bin(vehicle, (t + timedelta(seconds=i * step_s)).isoformat(),
                        soc + soc_drop_per_step, speed=max(speed, 0.0), **extra))
    return out


def test_capacity_table_contract():
    table = VEHICLE_CAPACITIES
    assert set(table) == {"ID1", "ID2", "CUP1", "CUP2", "CUP3", "CUP4", "CUP5"}
    for vc in table.values():
        assert vc.usable_capacity_kwh == DEFAULT_CAPACITY_KWH
        assert vc.assumption is True
        assert vc.provenance == "tum_dataset_readme_fleet_spec"
        assert vc.source_reference
    assert capacity_for("ID1") == 58.0
    assert capacity_table()[0]["contract_hash"]


def test_split_rules_deterministic():
    assert SPLIT_RULES == {"train": ["CUP1", "CUP2", "CUP4", "CUP5"],
                           "val": ["ID1"], "test": ["ID2", "CUP3"]}
    assert assign_split("CUP1") == "train"
    assert assign_split("CUP5") == "train"
    assert assign_split("ID1") == "val"
    assert assign_split("ID2") == "test"
    assert assign_split("CUP3") == "test"
    assert assign_split("CUP2") == "train"


def test_split_manifest_never_splits_episodes():
    bins = (
        _run("CUP1", "2022-12-12T10:00:00", 30, soc_drop_per_step=0.01)
        + _run("CUP1", "2022-12-12T11:00:00", 30, soc_drop_per_step=0.01)
        + _run("ID1", "2022-12-12T08:00:00", 30, soc_drop_per_step=0.01)
    )
    from calibration.windowing import build_windows  # noqa: F811
    w = build_windows(bins)
    assert len(w) == 3  # three distinct runs
    runs = {x.run_id for x in w}
    assert len(runs) == 3
    manifest = build_manifest(w)
    by_run = {m.run_id: m.split for m in manifest}
    assert len(by_run) == 3
    # whole episode -> single split, no episode spans splits
    assert assign_split("CUP1") == "train"
    assert all(by_run[r] == "train" for r in by_run if r.startswith("CUP1"))
    assert all(by_run[r] == "val" for r in by_run if r.startswith("ID1"))


def test_soc_drop_convention_target():
    from calibration.windowing import build_windows
    bins = _run("CUP1", "2022-12-12T10:00:00", 30, soc_drop_per_step=0.02)
    w = build_windows(bins)
    row = w[0]
    assert row.soc_start_pct > row.soc_end_pct
    assert row.observed_soc_drop_pct == pytest.approx(row.soc_start_pct - row.soc_end_pct)
    phys = PhysicsBaseline()
    cap = capacity_for("CUP1")
    from calibration.features import extract_features
    f = extract_features(row, phys, cap)
    assert f["observed_soc_drop_pct"] > 0
    assert f["physics_soc_drop_pct"] > 0
    assert f["residual_pct"] == pytest.approx(
        f["observed_soc_drop_pct"] - f["physics_soc_drop_pct"])


def test_physics_uses_measured_aux_no_hvac():
    bins = _run("CUP1", "2022-12-12T10:00:00", 30, soc_drop_per_step=0.01, aux=1500.0)
    phys = PhysicsBaseline()
    _, aux_wh, total = phys.energy_wh([b.signals for b in bins])
    assert aux_wh == pytest.approx(1500.0 * 300.0 / 3600.0 / 1.0)  # 125 Wh
    assert total == pytest.approx(aux_wh + phys.motive_wh([b.signals for b in bins]))
    # hvac never enters the baseline
    assert total > 0


def test_windowing_breaks_on_gaps():
    bins = (
        _run("CUP1", "2022-12-12T10:00:00", 30, soc_drop_per_step=0.01)
        + [   # gap of 600 s then a new run of 30
            _bin("CUP1", "2022-12-12T10:10:00", 60.0, 10.0)
        ] + _run("CUP1", "2022-12-12T10:10:10", 29, soc_drop_per_step=0.01)
    )
    runs = group_runs(bins, WindowConfig().max_gap_s)
    assert len(runs) == 2  # the isolated bin joins the second run via continuity
    assert len(runs[1]) == 30