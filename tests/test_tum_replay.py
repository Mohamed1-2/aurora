"""WS1 genuine TUM replay tests.

The replay path must read RAW long-format TUM parquet (time/value_id/value),
preserve original timestamps + vehicle identity + measured fields only +
chronological ordering, project each window through the existing canonical
mapping — and REFUSE to fabricate or fall back to the v2 JSONL corpus when the
raw parquet is absent (clean 503). A small synthetic parquet fixture exercises
the raw pipeline deterministically.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "telemetry-simulator"))
sys.path.insert(0, str(ROOT))

# Evict any cached foreign `app` package (repo runs several services' `app`
# packages in one process) so THIS test imports the telemetry-simulator package.
for _mod in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_mod, None)

pyarrow = pytest.importorskip("pyarrow")
pa = pytest.importorskip("pyarrow.parquet")

from app.engine.tum_replay import DataUnavailableError, TUMReplaySource

# value_id -> measured signal (must match SIGNAL_MAP in tum_driving.py)
_CORE = {
    4: "speed_kph",
    900: "soc_pct",
    15: "outside_temp_c",
    1209: "battery_temp_c",
    1269: "coolant_temp_c",
}
_EXTRA = {56: "hv_aux_power_w"}


def _write_raw_parquet(dir_path: Path, vehicle: str = "ID1", n_windows: int = 12) -> Path:
    """Write a synthetic RAW long-format TUM UDS parquet (measured fields only)."""
    rows = []
    t0 = dt.datetime(2022, 7, 17, 19, 10, 40, tzinfo=dt.timezone.utc).timestamp()
    for w in range(n_windows):
        base = t0 + w * 10
        rows.append((dt.datetime.fromtimestamp(base, tz=dt.timezone.utc), 4, 60 + w * 0.5))
        rows.append((dt.datetime.fromtimestamp(base + 1, tz=dt.timezone.utc), 900, 78 - w * 0.2))
        rows.append((dt.datetime.fromtimestamp(base + 2, tz=dt.timezone.utc), 15, 20.0))
        rows.append((dt.datetime.fromtimestamp(base + 3, tz=dt.timezone.utc), 1209, 28.0))
        rows.append((dt.datetime.fromtimestamp(base + 4, tz=dt.timezone.utc), 1269, 55.0))
        rows.append((dt.datetime.fromtimestamp(base + 5, tz=dt.timezone.utc), 56, 800.0))
    lib = __import__("pyarrow").table(
        {"time": [r[0] for r in rows], "value_id": [r[1] for r in rows], "value": [r[2] for r in rows]}
    )
    out = Path(dir_path) / f"{vehicle}.parquet"
    pa.write_table(lib, str(out))
    return out


@pytest.fixture(name="uds_dir")
def _uds_dir(tmp_path):
    _write_raw_parquet(tmp_path)
    return str(tmp_path)


def test_replay_available_and_episodes(uds_dir):
    src = TUMReplaySource(uds_dir)
    assert src.available() is True
    desc = src.describe()
    assert desc["vehicles"] == ["ID1"]
    eps = src.episodes("ID1")
    assert len(eps) == 1
    assert eps[0]["episode_id"] == "ID1-2022-07-17"
    assert "hv_aux_power_w" in eps[0]["signals_measured"]


def test_replay_preserves_timestamps_chronology_and_measured_only(uds_dir):
    src = TUMReplaySource(uds_dir)
    frames = src.slice("ID1", None, 0, 5)
    assert len(frames) == 5
    stamps = [f["window_start_utc"] for f in frames]
    assert stamps == sorted(stamps)
    assert stamps[0] == "2022-07-17T19:10:40+00:00"
    # Measured fields only — no fabricated destination/HVAC/cabin.
    f0 = frames[0]
    assert "destination" not in f0["signals"]
    assert "hvac" not in f0["signals"]
    assert "cabin_set_temp" not in f0["signals"]
    assert "speed_kph" in f0["signals"]
    assert "hv_aux_power_w" in f0["signals"]  # measured aux preserved
    # Provenance preserved.
    assert f0["provenance"]["source_dataset"] == "TUM_UDS"
    assert f0["provenance"]["vehicle_id"] == "ID1"


def test_replay_canonical_projection(uds_dir):
    src = TUMReplaySource(uds_dir)
    f0 = src.slice("ID1", None, 0, 1)[0]
    # Existing canonical mapping projects measured subset (no fabrication).
    for key in ("speed_kph", "soc_pct", "outside_temp_c", "battery_temp_c", "coolant_temp_c"):
        assert key in f0["canonical"]
    assert f0["canonical"]["soc_pct"] == pytest.approx(78.0, abs=1e-6)


def test_replay_missing_parquet_raises_clean_error(tmp_path):
    src = TUMReplaySource(str(tmp_path / "does-not-exist"))
    assert src.available() is False
    with pytest.raises(DataUnavailableError, match="refusing to fabricate"):
        src.slice("ID1", None, 0, 5)
    # describe() returns structured error, not a fabricated replay.
    d = src.describe()
    assert d["available"] is False
    assert "refusing to fabricate" in d["error"]


def test_replay_no_corpus_fallback(uds_dir, tmp_path):
    """Even when raw parquet is absent the source must NOT fall back to the v2
    JSONL corpus (a training/eval artifact, not a telemetry source)."""
    src = TUMReplaySource(str(tmp_path / "empty"))
    with pytest.raises(DataUnavailableError):
        src.slice("ID1", None, 0, 5)


def test_replay_slice_window_subset(uds_dir):
    src = TUMReplaySource(uds_dir)
    full = src.slice("ID1", None, 0, None)
    # default (no episode filter) selects longest episode = all 12 windows
    assert len(full) == 12
    sub = src.slice("ID1", None, 3, 4)
    assert len(sub) == 4
    assert sub[0]["window_start_utc"] == full[3]["window_start_utc"]