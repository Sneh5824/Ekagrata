"""Threaded camera capture: stamps each frame right after retrieve() and passes it on through a bounded queue.

Timestamp point (`t_host_ns`, SPEC §4.1, since 2026-09-30): "after_retrieve" = the frame's ARRIVAL at the
host plus JPEG decode (≈ 3 ms measured for 1280x720), NOT the exposure time. With OpenCV's DirectShow backend
grab() returns immediately and retrieve() blocks until the next frame arrives, so the older "after_grab"
stamp was ≈ one frame interval early. The unknown camera-internal latency (exposure, readout, USB, driver) is
what the LED sync (M7) and experiment E07 measure. `t_grab_ns` (after grab()) is kept as a diagnostic only.

The queue drops the OLDEST frame when full, so a slow consumer always gets recent frames; every drop is
counted and reported per delivered frame as `dropped_before`.

Wrong-camera guard (start()): the driver-reported fourcc must equal the requested one, and the first
STARTUP_CHECK_FRAMES frames must not all be near-black (a virtual camera such as Iriun shows a black
"waiting" screen and can take index 0).
"""

import threading
from collections import deque

import cv2

from ekagrata.core.config import CameraConfig
from ekagrata.core.timebase import now_ns
from ekagrata.core.types import CameraFrame

TIMESTAMP_POINT = "after_retrieve"
_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
_MAX_CONSECUTIVE_FAILURES = 50
STARTUP_CHECK_FRAMES = 5  # frames read (and discarded) in start() for the near-black check
BLACK_MEAN_MAX = 5.0  # a frame with mean pixel value (0-255) at or below this counts as near-black


def fourcc_to_str(code: float) -> str:
    code = int(code)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def describe(cfg: CameraConfig) -> str:
    """One startup line naming the device index and backend (printed by every camera-using script)."""
    return f"Camera: device index {cfg.device}, backend {cfg.backend}"


class CameraCapture:
    """Capture thread around cv2.VideoCapture. Use start() / get() / stop()."""

    def __init__(self, cfg: CameraConfig, capture_factory=cv2.VideoCapture):
        self.cfg = cfg
        self._factory = capture_factory
        self._cap = None
        self._queue: deque = deque()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_delivered_idx = -1
        self.captured = 0  # frames grabbed successfully
        self.dropped = 0  # frames discarded because the queue was full
        self.t_first_ns: int | None = None
        self.t_last_ns: int | None = None
        self.error: str | None = None

    def start(self) -> dict:
        """Open the camera, apply requested settings, check it is the intended camera, start the thread.

        Returns the requested and the driver-REPORTED settings (not measured). Raises RuntimeError if the
        camera cannot be opened, reports a different fourcc than requested, or delivers only near-black
        frames."""
        cfg = self.cfg
        cap = self._factory(cfg.device, _BACKENDS[cfg.backend])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open camera index {cfg.device} with backend '{cfg.backend}'")
        fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Some DirectShow drivers only accept the codec after the resolution is set.
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if cfg.exposure_auto is not None:
            # DSHOW convention: 0.75 = auto on, 0.25 = manual.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if cfg.exposure_auto else 0.25)
        if cfg.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
        info = {
            "device": cfg.device,
            "backend": cfg.backend,
            "timestamp_point": TIMESTAMP_POINT,
            "requested": {"w": cfg.width, "h": cfg.height, "fps": cfg.fps, "fourcc": cfg.fourcc},
            "reported": {
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
                "auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
                "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
            },
        }
        try:
            self._check_intended_camera(cap, info["reported"]["fourcc"])
        except RuntimeError:
            cap.release()
            raise
        self._cap = cap
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        return info

    def _check_intended_camera(self, cap, reported_fourcc: str) -> None:
        cfg = self.cfg
        hint = ("It is probably not the intended camera (e.g. the Iriun virtual camera can take index 0). "
                "Try another index with --camera N, or set `device` in configs/camera.yaml.")
        if reported_fourcc != cfg.fourcc:
            raise RuntimeError(
                f"camera index {cfg.device} ({cfg.backend}): requested fourcc {cfg.fourcc!r} but the driver "
                f"reports {reported_fourcc!r}. {hint}")
        means, attempts = [], 0
        while len(means) < STARTUP_CHECK_FRAMES and attempts < _MAX_CONSECUTIVE_FAILURES:
            attempts += 1
            if not cap.grab():
                continue
            ok, image = cap.retrieve()
            if ok and image is not None:
                means.append(float(image.mean()))
        if not means:
            raise RuntimeError(
                f"camera index {cfg.device} ({cfg.backend}) delivered no frames at startup. {hint}")
        if max(means) <= BLACK_MEAN_MAX:
            raise RuntimeError(
                f"camera index {cfg.device} ({cfg.backend}): the first {len(means)} frames are near-black "
                f"(mean pixel value <= {BLACK_MEAN_MAX}; max {max(means):.1f}). {hint}")

    def _run(self) -> None:
        failures = 0
        frame_idx = 0
        while not self._stop.is_set():
            if not self._cap.grab():
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    self.error = f"camera stopped delivering frames ({failures} consecutive grab failures)"
                    break
                continue
            t_grab_ns = now_ns()  # diagnostic only: on DirectShow grab() returns before the frame arrives
            ok, image = self._cap.retrieve()
            # after_retrieve: frame ARRIVAL at the host + decode (~3 ms), not exposure time (see module doc).
            t_host_ns = now_ns()
            if not ok or image is None:
                failures += 1
                continue
            failures = 0
            with self._cond:
                if self.t_first_ns is None:
                    self.t_first_ns = t_host_ns
                self.t_last_ns = t_host_ns
                self.captured += 1
                if len(self._queue) >= self.cfg.queue_size:
                    self._queue.popleft()
                    self.dropped += 1
                self._queue.append((frame_idx, t_host_ns, image, t_grab_ns))
                self._cond.notify()
            frame_idx += 1
        with self._cond:
            self._cond.notify_all()

    def get(self, timeout_s: float = 1.0) -> CameraFrame | None:
        """Next frame, or None if none arrived within the timeout (check `alive` / `error`)."""
        with self._cond:
            if not self._queue:
                self._cond.wait(timeout_s)
            if not self._queue:
                return None
            frame_idx, t_host_ns, image, t_grab_ns = self._queue.popleft()
        dropped_before = frame_idx - self._last_delivered_idx - 1
        self._last_delivered_idx = frame_idx
        return CameraFrame(frame_idx, t_host_ns, None, dropped_before, image, t_grab_ns)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def measured_fps(self) -> float | None:
        """Capture rate measured from frame timestamps, or None with fewer than 2 frames."""
        if self.captured < 2 or self.t_last_ns == self.t_first_ns:
            return None
        return (self.captured - 1) / ((self.t_last_ns - self.t_first_ns) * 1e-9)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
