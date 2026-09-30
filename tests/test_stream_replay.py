"""Replay path of scripts/stream_to_blender.py on a synthetic (SIMULATED) landmarks parquet."""

import importlib.util
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ekagrata.io.session import _LANDMARK_SCHEMA, LANDMARKS_PARQUET, landmark_columns
from ekagrata.transport.udp_twin import decode

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stream_to_blender", REPO / "scripts" / "stream_to_blender.py")
stream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stream)

FRAME_NS = 33_333_333  # 30 Hz


def synthetic_session(tmp_path, n=31, no_pose=(5, 6)) -> tuple[Path, np.ndarray]:
    """SIMULATED landmarks table in the real session schema; frames in `no_pose` have no detection."""
    t = 10**12 + np.arange(n, dtype=np.int64) * FRAME_NS
    rows = {c: np.full(n, np.nan) for c in landmark_columns()}
    rows["frame_idx"] = np.arange(n, dtype=np.int64)
    rows["t_sync_ns"] = t
    for i in range(33):
        rows[f"lm{i}_vis"] = np.full(n, 0.9)
        for k, v in zip("xyz", (0.01 * i, -0.02 * i, 0.03), strict=True):
            rows[f"wlm{i}_{k}"] = np.full(n, v)
    for j in no_pose:
        for i in range(33):
            rows[f"lm{i}_vis"][j] = np.nan
            for k in "xyz":
                rows[f"wlm{i}_{k}"][j] = np.nan
    d = tmp_path / "2026-09-30_synthetic_sim"
    (d / LANDMARKS_PARQUET).parent.mkdir(parents=True)
    table = pa.Table.from_pandas(pd.DataFrame(rows)[landmark_columns()], schema=_LANDMARK_SCHEMA,
                                 preserve_index=False)
    pq.write_table(table, d / LANDMARKS_PARQUET)
    return d, t


class FakeTime:
    """Deterministic clock: sleep(s) advances time by s (at least 10 us, so a yield also makes progress)."""

    def __init__(self, t0=5_000_000_000):
        self.t = t0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += max(int(s * 1e9), 10_000)


def test_load_replay(tmp_path):
    d, t = synthetic_session(tmp_path)
    t2, world, vis = stream.load_replay(d)
    assert np.array_equal(t2, t) and world.shape == (31, 33, 3) and vis.shape == (31, 33)
    assert np.isnan(world[5]).all() and np.isfinite(world[4]).all()


def test_load_replay_rejects_missing_and_non_monotonic(tmp_path):
    with pytest.raises(FileNotFoundError):
        stream.load_replay(tmp_path)
    d, _ = synthetic_session(tmp_path)
    df = pd.read_parquet(d / LANDMARKS_PARQUET)
    df.loc[3, "t_sync_ns"] = df.loc[2, "t_sync_ns"]
    df.to_parquet(d / LANDMARKS_PARQUET)
    with pytest.raises(ValueError, match="strictly increasing"):
        stream.load_replay(d)


def test_scheduler_fake_clock_exact():
    """Authoritative correctness test: every emit happens at start + (t_i - t_0), never early."""
    ft = FakeTime()
    t = 10**12 + np.array([0, 33_000_000, 70_000_000, 100_000_000, 140_000_000], dtype=np.int64)
    emitted = []
    start = ft.t
    late = stream.run_replay(t, lambda i: emitted.append((i, ft.t)), clock=ft.clock, sleep=ft.sleep)
    assert [i for i, _ in emitted] == [0, 1, 2, 3, 4]
    for (i, at), lat in zip(emitted, late, strict=True):
        target = start + int(t[i] - t[0])
        assert 0 <= at - target <= 10_000 and lat == at - target


def test_scheduler_loop_and_stop():
    ft = FakeTime()
    t = 10**12 + np.arange(4, dtype=np.int64) * FRAME_NS
    emitted = []
    start = ft.t
    stream.run_replay(t, lambda i: emitted.append((i, ft.t)), clock=ft.clock, sleep=ft.sleep, loop=True,
                      max_passes=2)
    assert [i for i, _ in emitted] == [0, 1, 2, 3, 0, 1, 2, 3]
    second_start = start + 3 * FRAME_NS + FRAME_NS  # one median interval after the last frame
    assert 0 <= emitted[4][1] - second_start <= 10_000
    calls = []
    stream.run_replay(t, calls.append, clock=ft.clock, sleep=ft.sleep, loop=True,
                      should_stop=lambda: len(calls) >= 6)
    assert calls == [0, 1, 2, 3, 0, 1]


@pytest.mark.timing
def test_scheduler_real_clock_timing():
    """Real clock, 1 s at 30 Hz. Tolerance: never early by > 1 ms, median lateness <= 5 ms, max <= 30 ms,
    and the emitted span within +/-30 ms of the recorded span."""
    t = 10**12 + np.arange(31, dtype=np.int64) * FRAME_NS
    times = []
    start = time.perf_counter_ns()
    late = stream.run_replay(t, lambda i: times.append(time.perf_counter_ns()))
    late_ms = np.asarray(late, dtype=np.float64) * 1e-6
    targets = start + (t - t[0])
    own_ms = (np.asarray(times) - targets) * 1e-6  # independent of run_replay's own bookkeeping
    span_err_ms = ((times[-1] - times[0]) - (t[-1] - t[0])) * 1e-6
    report = (f"emit-time lateness min {own_ms.min():.3f} ms; "
              f"scheduler lateness median {np.median(late_ms):.3f}, "
              f"p95 {np.percentile(late_ms, 95):.3f}, max {late_ms.max():.3f}; "
              f"span error {span_err_ms:+.3f} ms")
    print(report)
    assert own_ms.min() >= -1.0, report
    assert np.median(late_ms) <= 5.0, report
    assert late_ms.max() <= 30.0, report
    assert abs(span_err_ms) <= 30.0, report


def test_main_replay_end_to_end(tmp_path, capsys):
    """Replay a synthetic session through the real sender to a real loopback listener."""
    d, t = synthetic_session(tmp_path, n=16)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    cfg = tmp_path / "blender.yaml"
    cfg.write_text(
        'blender_exe: "unused.exe"\nblender_version: "5.0.1"\nudp_host: 127.0.0.1\n'
        f"udp_port: {rx.getsockname()[1]}\nsend_rate_max_hz: 1000\nhitting_side: right\n"
        "landmarks: [r_shoulder, r_wrist]\nplacement_offset_m: [0, 0, 0]\n"
        "matched_camera: {distance_m: 2.5, height_m: 0.3, yaw_deg: 0, horizontal_fov_deg: 70}\n",
        encoding="utf-8")
    rc = stream.main(["--session", str(d), "--config", str(cfg), "--camera-config",
                      str(REPO / "configs" / "camera.yaml")])
    msgs = [decode(rx.recv(65535)) for _ in range(16)]
    rx.close()
    out = capsys.readouterr().out
    assert rc == 0
    assert [m["seq"] for m in msgs] == list(range(1, 17))
    assert [m["t_sync_ns"] for m in msgs] == t.tolist()
    assert all(m["source"] == "replay" for m in msgs)
    assert set(msgs[0]["points"]) == {"r_shoulder", "r_wrist"} and msgs[5]["points"] == {}
    assert "Messages sent     : 16" in out and "Frames with pose  : 14" in out
    assert "axis mapping unverified until check_axes passes" in out


def test_stage_durations_known_answers():
    """Latency breakdown arithmetic on hand-made stamps (ns): each stage = difference of its two stamps."""
    rows = []
    for i in range(3):
        g = i * 33_000_000
        rows.append({"t_grab_ns": g, "t_host_ns": g + 4_000_000, "t_dequeue_ns": g + 10_000_000,
                     "t_infer_start_ns": g + 10_100_000, "t_infer_end_ns": g + 40_100_000,
                     "t_send_ns": g + 40_300_000, "t_sent_ns": g + 40_400_000})
    d = stream.stage_durations_ms(rows)
    assert np.allclose(d["grab -> retrieve end"], 4.0) and np.allclose(d["queue wait"], 6.0)
    assert np.allclose(d["dequeue -> inference"], 0.1) and np.allclose(d["inference"], 30.0)
    assert np.allclose(d["map + smooth"], 0.2) and np.allclose(d["encode + sendto"], 0.1)
    assert np.allclose(d["t_host -> sent"], 36.4)
    report = "\n".join(stream.stage_report(rows))
    assert "queue wait / frame interval: median 0.18" in report  # 6 ms / 33 ms
    assert stream.stage_report([]) == ["(no messages sent: no latency rows)"]


def test_record_and_preview_argument_rules():
    with pytest.raises(SystemExit):
        stream.parse_args(["--record", "run1"])  # needs --athlete
    with pytest.raises(SystemExit):
        stream.parse_args(["--session", "x", "--record", "run1", "--athlete", "002"])  # live only
    with pytest.raises(SystemExit):
        stream.parse_args(["--preview-geometry", "0", "0", "640", "360"])  # needs --preview
    args = stream.parse_args(["--preview", "--preview-geometry", "0", "0", "640", "360", "--record", "run1",
                              "--athlete", "002"])
    assert args.preview_geometry == [0, 0, 640, 360] and args.record == "run1"
    assert stream.PREVIEW_WINDOW == "EKAGRATA camera"
