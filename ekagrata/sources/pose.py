"""Frame sources for the pose pipeline. Live camera and video file produce identical `CameraFrame`s."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ekagrata.core.config import CameraConfig
from ekagrata.core.types import CameraFrame
from ekagrata.io.camera import CameraCapture, fourcc_to_str


class LiveCameraSource:
    """Live camera frames, timestamped on the host clock right after grab."""

    def __init__(self, cfg: CameraConfig, capture_factory=cv2.VideoCapture):
        self.capture = CameraCapture(cfg, capture_factory)
        self._info = self.capture.start()

    def info(self) -> dict:
        return self._info

    def frames(self) -> Iterator[CameraFrame]:
        while True:
            frame = self.capture.get(timeout_s=1.0)
            if frame is not None:
                yield frame
            elif not self.capture.alive:
                if self.capture.error:
                    raise RuntimeError(self.capture.error)
                return

    def close(self) -> None:
        self.capture.stop()


class VideoFileSource:
    """Frames from a video file. `pts_ns` comes from the container timestamps (CAP_PROP_POS_MSEC), measured
    from the start of the file; `t_host_ns` is set equal to `pts_ns`. No frames are dropped."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video file: {self.path}")

    def info(self) -> dict:
        cap = self._cap
        return {
            "path": str(self.path),
            "reported": {
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),  # container-declared, not measured
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            },
        }

    def frames(self) -> Iterator[CameraFrame]:
        frame_idx = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            pts_ns = int(round(self._cap.get(cv2.CAP_PROP_POS_MSEC) * 1_000_000))
            yield CameraFrame(frame_idx, pts_ns, pts_ns, 0, image)
            frame_idx += 1

    def close(self) -> None:
        self._cap.release()


def pts_report(pts_ns) -> dict:
    """Measured timestamp spacing of a frame sequence. `irregular` counts intervals deviating from the median
    by more than 25%; `non_increasing` counts intervals <= 0."""
    t = np.asarray(pts_ns, dtype=np.int64)
    if t.size < 2:
        return {"n_frames": int(t.size)}
    dt_ms = np.diff(t).astype(np.float64) * 1e-6
    median = float(np.median(dt_ms))
    span_s = (t[-1] - t[0]) * 1e-9
    return {
        "n_frames": int(t.size),
        "span_s": float(span_s),
        "measured_fps": float((t.size - 1) / span_s) if span_s > 0 else None,
        "interval_ms_median": median,
        "interval_ms_mean": float(dt_ms.mean()),
        "interval_ms_min": float(dt_ms.min()),
        "interval_ms_max": float(dt_ms.max()),
        "non_increasing": int(np.sum(dt_ms <= 0)),
        "irregular": int(np.sum(np.abs(dt_ms - median) > 0.25 * median)),
    }
