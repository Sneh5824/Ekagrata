import threading
import time

import numpy as np
import pytest

from ekagrata.core.config import CameraConfig
from ekagrata.io.camera import STARTUP_CHECK_FRAMES, CameraCapture, fourcc_to_str
from ekagrata.sources.pose import LiveCameraSource


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture: `startup_frames` frames of value `startup_value` (consumed by
    the wrong-camera guard in start()), then `n_frames` data frames of value 1, 2, ..., `period_s` apart.
    `reported_fourcc` overrides what the driver reports (default: echo the requested fourcc)."""

    def __init__(self, n_frames=20, period_s=0.002, opened=True, startup_frames=STARTUP_CHECK_FRAMES,
                 startup_value=128, reported_fourcc=None):
        self.n_frames = n_frames
        self.period_s = period_s
        self.opened = opened
        self.startup_left = startup_frames
        self.startup_value = startup_value
        self.reported_fourcc = reported_fourcc
        self.in_startup = False
        self.i = 0
        self.props = {}
        self.released = False

    def isOpened(self):  # noqa: N802 (OpenCV API name)
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FOURCC and self.reported_fourcc is not None:
            return float(cv2.VideoWriter_fourcc(*self.reported_fourcc))
        return float(self.props.get(prop, 0.0))

    def grab(self):
        if self.startup_left > 0:
            self.startup_left -= 1
            self.in_startup = True
            return True
        self.in_startup = False
        if self.i >= self.n_frames:
            time.sleep(0.001)
            return False
        time.sleep(self.period_s)
        self.i += 1
        return True

    def retrieve(self):
        value = self.startup_value if self.in_startup else self.i
        return True, np.full((4, 6, 3), value, dtype=np.uint8)

    def release(self):
        self.released = True


def cfg(queue_size=4):
    return CameraConfig(device=0, backend="any", width=6, height=4, fps=30, fourcc="MJPG",
                        model_variant="lite", queue_size=queue_size,
                        mp_to_world=((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))


def factory_for(fake):
    return lambda device, api: fake


def test_fourcc_to_str():
    import cv2

    assert fourcc_to_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"


def test_open_failure_is_clear():
    cap = CameraCapture(cfg(), factory_for(FakeCapture(opened=False)))
    with pytest.raises(RuntimeError, match="cannot open camera index 0"):
        cap.start()


def test_all_frames_delivered_with_increasing_timestamps():
    fake = FakeCapture(n_frames=20)
    cap = CameraCapture(cfg(queue_size=100), factory_for(fake))
    info = cap.start()
    assert info["requested"]["fps"] == 30 and "reported" in info
    frames = []
    while len(frames) < 20:
        f = cap.get(timeout_s=2.0)
        assert f is not None
        frames.append(f)
    cap.stop()
    assert [f.frame_idx for f in frames] == list(range(20))
    t = np.array([f.t_host_ns for f in frames])
    assert np.all(np.diff(t) > 0)
    assert all(f.dropped_before == 0 and f.pts_ns is None for f in frames)
    assert all(f.t_grab_ns is not None and f.t_grab_ns <= f.t_host_ns for f in frames)
    assert cap.captured == 20 and cap.dropped == 0
    assert cap.measured_fps() is not None and cap.measured_fps() > 0
    assert fake.released


def test_drop_oldest_counts_and_reports_gaps():
    fake = FakeCapture(n_frames=30, period_s=0.001)
    cap = CameraCapture(cfg(queue_size=2), factory_for(fake))
    cap.start()
    # Let the producer finish while nobody consumes: only the newest 2 frames survive.
    deadline = time.monotonic() + 5.0
    while cap.captured < 30 and time.monotonic() < deadline:
        time.sleep(0.01)
    first = cap.get(timeout_s=1.0)
    second = cap.get(timeout_s=1.0)
    cap.stop()
    assert cap.captured == 30
    assert cap.dropped == 28
    assert (first.frame_idx, second.frame_idx) == (28, 29)
    assert first.dropped_before == 28  # frames 0..27 never delivered
    assert second.dropped_before == 0
    assert np.all(first.image == 29)  # image of the 29th grab (FakeCapture counts from 1)


def test_get_times_out_without_frames():
    cap = CameraCapture(cfg(), factory_for(FakeCapture(n_frames=0)))
    cap.start()  # startup frames pass the guard; then no data frames arrive
    assert cap.get(timeout_s=0.05) is None
    cap.stop()


def test_consecutive_failures_stop_thread_with_error():
    cap = CameraCapture(cfg(), factory_for(FakeCapture(n_frames=0)))
    cap.start()
    deadline = time.monotonic() + 5.0
    while cap.alive and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not cap.alive
    assert "grab failures" in cap.error
    cap.stop()


def test_live_source_iterates_and_closes():
    fake = FakeCapture(n_frames=10)
    src = LiveCameraSource(cfg(queue_size=100), capture_factory=factory_for(fake))
    got = []
    with pytest.raises(RuntimeError, match="grab failures"):
        for f in src.frames():
            got.append(f.frame_idx)
    src.close()
    assert got == list(range(10))
    assert threading.active_count() >= 1


class BlockingRetrieveCapture(FakeCapture):
    """DirectShow-like: grab() returns at once, retrieve() blocks for one frame interval. Records when each
    data frame's retrieve() finished."""

    def __init__(self, n_frames, interval_s=0.03):
        super().__init__(n_frames=n_frames, period_s=0.0)
        self.interval_s = interval_s
        self.retrieve_end_ns = []
        self.grab_end_ns = []

    def grab(self):
        ok = super().grab()
        if ok and not self.in_startup:
            self.grab_end_ns.append(time.perf_counter_ns())
        return ok

    def retrieve(self):
        if not self.in_startup:
            time.sleep(self.interval_s)
            out = super().retrieve()
            self.retrieve_end_ns.append(time.perf_counter_ns())
            return out
        return super().retrieve()


def test_timestamp_is_taken_after_retrieve():
    fake = BlockingRetrieveCapture(n_frames=5)
    cap = CameraCapture(cfg(queue_size=100), factory_for(fake))
    cap.start()
    frames = [cap.get(timeout_s=2.0) for _ in range(5)]
    cap.stop()
    for i, f in enumerate(frames):
        assert f.frame_idx == i
        assert f.t_host_ns >= fake.retrieve_end_ns[i]  # after the blocking retrieve(), not after grab()
        assert f.t_host_ns - f.t_grab_ns >= 0.03e9 * 0.9  # the one-interval wait sits between the two stamps
        assert f.t_grab_ns >= fake.grab_end_ns[i]


def test_info_reports_timestamp_point():
    cap = CameraCapture(cfg(), factory_for(FakeCapture(n_frames=1)))
    info = cap.start()
    cap.stop()
    assert info["timestamp_point"] == "after_retrieve"


def test_fourcc_mismatch_fails_naming_both():
    fake = FakeCapture(reported_fourcc="YUY2")
    cap = CameraCapture(cfg(), factory_for(fake))
    match = r"requested fourcc 'MJPG' but the driver reports 'YUY2'.*--camera"
    with pytest.raises(RuntimeError, match=match):
        cap.start()
    assert fake.released


def test_near_black_startup_frames_fail_with_hint():
    fake = FakeCapture(startup_value=1)
    cap = CameraCapture(cfg(), factory_for(fake))
    with pytest.raises(RuntimeError, match=r"near-black.*Try another index"):
        cap.start()
    assert fake.released


def test_no_frames_at_startup_fails():
    fake = FakeCapture(n_frames=0, startup_frames=0)
    cap = CameraCapture(cfg(), factory_for(fake))
    with pytest.raises(RuntimeError, match="no frames at startup"):
        cap.start()
