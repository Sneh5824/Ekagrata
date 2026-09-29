import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from ekagrata.core.types import CameraFrame, PoseFrame
from ekagrata.io.session import (
    FRAME_COLUMNS,
    SessionWriter,
    base_metadata,
    create_session,
    landmark_columns,
    read_session,
    session_dir_name,
)

SPEC_43_KEYS = {
    "schema_version", "session_id", "athlete_id", "created_utc", "handedness", "code_version", "host",
    "camera", "model", "imu", "calibration", "clock_sync", "notes", "derived",
}


def make_pair(i, rng, detected=True, pts=None):
    frame = CameraFrame(i, 1_000_000_000 + i * 33_333_333, pts, i % 3, np.zeros((2, 2, 3), np.uint8))
    img = rng.random((33, 5)) if detected else np.full((33, 5), np.nan)
    world = rng.normal(size=(33, 3)) if detected else None
    return frame, PoseFrame(i, frame.t_host_ns, frame.t_host_ns, img, world)


def new_session(tmp_path, name="s1"):
    meta = base_metadata(name, "a01")
    return create_session(tmp_path, name, "a01", meta, date=dt.date(2026, 9, 29)), meta


def test_session_dir_name_and_validation():
    assert session_dir_name(dt.date(2026, 9, 29), "live01", "a01") == "2026-09-29_live01_a01"
    with pytest.raises(ValueError, match="session"):
        session_dir_name(dt.date(2026, 9, 29), "live_01", "a01")  # '_' is the field separator
    with pytest.raises(ValueError, match="athlete"):
        session_dir_name(dt.date(2026, 9, 29), "live01", "a 01")


def test_metadata_has_all_spec_keys():
    meta = base_metadata("s1", "a01")
    assert SPEC_43_KEYS <= set(meta)
    assert meta["code_version"]  # git describe or "unknown (no git)"


def test_create_session_never_overwrites(tmp_path):
    path, _ = new_session(tmp_path)
    assert path.name == "2026-09-29_s1_a01"
    assert (path / "session.json").is_file()
    with pytest.raises(FileExistsError, match="append-only"):
        create_session(tmp_path, "s1", "a01", {}, date=dt.date(2026, 9, 29))


def test_write_read_round_trip(tmp_path):
    path, meta = new_session(tmp_path)
    rng = np.random.default_rng(0)
    pairs = [make_pair(i, rng, detected=(i % 4 != 0), pts=(i * 4_166_667 if i % 2 else None))
             for i in range(250)]  # > flush_every: several Parquet row groups
    with SessionWriter(path, flush_every=100) as w:
        for frame, pose in pairs:
            w.add(frame, pose)
        w.finalize({"summary": {"frames_processed": 250}})

    meta2, frames, lms = read_session(path)
    for key in SPEC_43_KEYS:
        assert meta2[key] == meta[key]
    assert meta2["summary"] == {"frames_processed": 250} and meta2["n_frames"] == 250
    assert "aborted" not in meta2

    assert list(frames.columns) == FRAME_COLUMNS
    assert frames["frame_idx"].tolist() == list(range(250))
    assert frames["t_host_ns"].tolist() == [f.t_host_ns for f, _ in pairs]
    assert frames["dropped_before"].tolist() == [f.dropped_before for f, _ in pairs]
    expected_pts = pd.array([f.pts_ns for f, _ in pairs], dtype="Int64")
    assert frames["pts_ns"].equals(pd.Series(expected_pts, name="pts_ns"))
    assert np.all(np.diff(frames["t_sync_ns"].to_numpy()) > 0)

    assert list(lms.columns) == landmark_columns()
    assert len(landmark_columns()) == 2 + 33 * 5 + 33 * 3
    assert lms["frame_idx"].dtype == np.int64 and lms["t_sync_ns"].dtype == np.int64
    for (_, pose), (_, row) in zip(pairs, lms.iterrows(), strict=True):
        img = row[[f"lm{i}_{k}" for i in range(33) for k in ("x", "y", "z", "vis", "pres")]].to_numpy(float)
        wld = row[[f"wlm{i}_{k}" for i in range(33) for k in ("x", "y", "z")]].to_numpy(float)
        np.testing.assert_array_equal(img.reshape(33, 5), pose.landmarks_img)
        if pose.landmarks_world is None:
            assert np.all(np.isnan(wld))
        else:
            np.testing.assert_array_equal(wld.reshape(33, 3), pose.landmarks_world)


def test_exception_still_finalises_readable_files(tmp_path):
    path, _ = new_session(tmp_path)
    rng = np.random.default_rng(1)
    with pytest.raises(KeyboardInterrupt), SessionWriter(path, flush_every=100) as w:
        for i in range(150):
            w.add(*make_pair(i, rng))
        raise KeyboardInterrupt
    meta, frames, lms = read_session(path)
    assert meta["aborted"] == "KeyboardInterrupt"
    assert len(frames) == 150 and len(lms) == 150  # nothing lost, including the unflushed tail


def test_empty_session_is_readable(tmp_path):
    path, _ = new_session(tmp_path)
    with SessionWriter(path) as w:
        w.finalize()
    meta, frames, lms = read_session(path)
    assert meta["n_frames"] == 0 and len(frames) == 0 and len(lms) == 0
    assert list(lms.columns) == landmark_columns()


def test_non_ascii_notes_round_trip(tmp_path):
    notes = "एकाग्रता smash — 90° elbow, ±2000 °/s, π/2, naïve ✓"
    meta = base_metadata("s1", "a01")
    meta["notes"] = notes
    path = create_session(tmp_path, "s1", "a01", meta, date=dt.date(2026, 9, 29))
    rng = np.random.default_rng(2)
    with SessionWriter(path) as w:
        w.add(*make_pair(0, rng))
        w.finalize({"summary": {"stop_reason": "Ctrl+C — user"}})
    meta2, frames, _ = read_session(path)
    assert meta2["notes"] == notes
    assert meta2["summary"]["stop_reason"] == "Ctrl+C — user"
    assert len(frames) == 1
    # The file on disk is valid UTF-8 regardless of the Windows locale code page.
    json.loads((path / "session.json").read_bytes().decode("utf-8"))


def test_session_json_is_valid_json(tmp_path):
    path, _ = new_session(tmp_path)
    json.loads((path / "session.json").read_text(encoding="utf-8"))
