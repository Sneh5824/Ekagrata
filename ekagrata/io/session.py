"""Session folders: data/sessions/<YYYY-MM-DD>_<session>_<athlete>/ (layout per SPEC §4.1).

Raw files are append-only: a session folder is never overwritten. `session.json` is written at start and
finalised at the end (SPEC §4.1).
"""

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ekagrata.core.types import CameraFrame, PoseFrame

SCHEMA_VERSION = 1
N_LANDMARKS = 33
FRAMES_CSV = Path("video") / "cam0_frames.csv"
LANDMARKS_PARQUET = Path("pose") / "cam0_landmarks.parquet"
FRAME_COLUMNS = ["frame_idx", "t_host_ns", "t_sync_ns", "pts_ns", "dropped_before"]
_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")
_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "mediapipe", "opencv-contrib-python", "pyyaml")


def landmark_columns() -> list[str]:
    """Parquet columns: frame_idx, t_sync_ns, lm{i}_{x,y,z,vis,pres} (image-normalised), wlm{i}_{x,y,z}
    (MediaPipe world, metres, raw MediaPipe axes)."""
    cols = ["frame_idx", "t_sync_ns"]
    cols += [f"lm{i}_{k}" for i in range(N_LANDMARKS) for k in ("x", "y", "z", "vis", "pres")]
    cols += [f"wlm{i}_{k}" for i in range(N_LANDMARKS) for k in ("x", "y", "z")]
    return cols


_LANDMARK_SCHEMA = pa.schema(
    [(c, pa.int64()) for c in landmark_columns()[:2]] + [(c, pa.float64()) for c in landmark_columns()[2:]]
)


def session_dir_name(date: dt.date, session: str, athlete: str) -> str:
    for label, value in (("session", session), ("athlete", athlete)):
        if not _NAME_RE.match(value):
            raise ValueError(f"{label} name {value!r} may only contain letters, digits and '-'")
    return f"{date.isoformat()}_{session}_{athlete}"


def code_version() -> str:
    """`git describe --always --dirty`, or 'unknown (no git)' when unavailable."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown (no git)"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown (no git)"


def host_info() -> dict:
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"os": platform.platform(), "python": sys.version.split()[0], "packages": packages}


def base_metadata(session_id: str, athlete_id: str, handedness: str = "right") -> dict:
    """session.json with every SPEC §4.3 key; keys owned by later milestones are None."""
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "athlete_id": athlete_id,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "handedness": handedness,
        "code_version": code_version(),
        "host": host_info(),
        "camera": None,
        "model": None,
        "imu": None,
        "calibration": None,
        "clock_sync": {"method": "none (M1: t_sync_ns = host clock, or video PTS in file mode)"},
        "notes": "",
        "derived": [],
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def create_session(root, session: str, athlete: str, meta: dict, date: dt.date | None = None) -> Path:
    """Create the session folder and write the initial session.json. Never overwrites an existing session."""
    name = session_dir_name(date or dt.date.today(), session, athlete)
    path = Path(root) / name
    if path.exists():
        raise FileExistsError(f"session folder already exists (raw data is append-only): {path}")
    (path / "video").mkdir(parents=True)
    (path / "pose").mkdir()
    write_json(path / "session.json", meta)
    return path


def pose_to_row(pose: PoseFrame) -> list:
    img = np.asarray(pose.landmarks_img, dtype=np.float64).reshape(N_LANDMARKS, 5)
    world = (np.full((N_LANDMARKS, 3), np.nan) if pose.landmarks_world is None
             else np.asarray(pose.landmarks_world, dtype=np.float64).reshape(N_LANDMARKS, 3))
    return [int(pose.frame_idx), int(pose.t_sync_ns), *img.ravel().tolist(), *world.ravel().tolist()]


class SessionWriter:
    """Buffered writer for cam0_frames.csv and cam0_landmarks.parquet. Use as a context manager so that
    Ctrl+C or an exception still finalises every file. The CSV is flushed every `flush_every` frames; the
    Parquet file gets one row group per `flush_every` frames (its footer is written at finalize)."""

    def __init__(self, session_dir, flush_every: int = 100):
        self.dir = Path(session_dir)
        self.flush_every = flush_every
        self._csv_file = open(self.dir / FRAMES_CSV, "x", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(FRAME_COLUMNS)
        self._pq = pq.ParquetWriter(self.dir / LANDMARKS_PARQUET, _LANDMARK_SCHEMA)
        self._rows: list[list] = []
        self.n_frames = 0
        self.finalized = False

    def add(self, frame: CameraFrame, pose: PoseFrame) -> None:
        self._csv.writerow([frame.frame_idx, frame.t_host_ns, pose.t_sync_ns,
                            "" if frame.pts_ns is None else frame.pts_ns, frame.dropped_before])
        self._rows.append(pose_to_row(pose))
        self.n_frames += 1
        if len(self._rows) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        self._csv_file.flush()
        if self._rows:
            columns = list(zip(*self._rows, strict=True))
            arrays = [pa.array(col, type=f.type) for col, f in zip(columns, _LANDMARK_SCHEMA, strict=True)]
            self._pq.write_table(pa.Table.from_arrays(arrays, schema=_LANDMARK_SCHEMA))
            self._rows.clear()

    def finalize(self, updates: dict | None = None) -> None:
        """Flush and close all files, then merge `updates` into session.json and mark it finalised."""
        if self.finalized:
            return
        try:
            self._flush()
        finally:
            self._pq.close()
            self._csv_file.close()
            self.finalized = True
        meta_path = self.dir / "session.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(updates or {})
        meta["finalized_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        meta["n_frames"] = self.n_frames
        write_json(meta_path, meta)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finalize({"aborted": exc_type.__name__} if exc_type else None)
        return False


def read_session(session_dir) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return (session.json dict, frames table, landmarks table)."""
    d = Path(session_dir)
    meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
    frames = pd.read_csv(d / FRAMES_CSV, dtype={"pts_ns": "Int64"})
    landmarks = pd.read_parquet(d / LANDMARKS_PARQUET)
    return meta, frames, landmarks
