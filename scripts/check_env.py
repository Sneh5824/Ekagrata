"""Print the environment and MEASURE the camera frame rate from perf_counter_ns timestamps.

By default the camera is opened exactly as the recorder does (configs/camera.yaml: device, backend,
resolution, fps request, fourcc, exposure; same CameraCapture code, including the wrong-camera guard).
--raw skips the config: it tries DSHOW then MSMF with only resolution/fps/MJPG requested (no exposure), as
before 2026-09-30.

Usage: uv run python scripts/check_env.py [--camera N] [--seconds 5]
       uv run python scripts/check_env.py --raw [--camera N] [--width 1280 --height 720 --fps 60]
"""

import argparse
import dataclasses
import importlib
import importlib.metadata
import platform
import sys
import time
from pathlib import Path

import numpy as np

WARMUP_FRAMES = 10


def print_versions() -> None:
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    for name in ("numpy", "scipy", "pandas", "cv2", "mediapipe"):
        try:
            mod = importlib.import_module(name)
            print(f"{name:<12}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:  # report and keep going
            print(f"{name:<12}: NOT AVAILABLE ({exc})")

    opencv_dists = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata["Name"] and d.metadata["Name"].lower().startswith("opencv")
    )
    print(f"OpenCV dists: {opencv_dists or 'none'}")
    if len(opencv_dists) > 1:
        print("WARNING: more than one OpenCV distribution is installed; keep exactly one.")


def report(stamps_ns, shape, brightness) -> int:
    """Print measured rate, interval statistics and mean brightness. Returns an exit code."""
    if len(stamps_ns) < 3:
        print(f"Too few frames captured ({len(stamps_ns)}) to measure fps.")
        return 1
    t = np.asarray(stamps_ns, dtype=np.int64)
    dt_ms = np.diff(t).astype(np.float64) * 1e-6
    median_ms = float(np.median(dt_ms))
    span_s = (t[-1] - t[0]) * 1e-9
    print(f"Frame shape : {shape}")
    print(f"Measured    : {len(dt_ms) / span_s:.2f} fps over {span_s:.2f} s ({len(t)} frames)")
    print(
        f"Interval ms : mean {dt_ms.mean():.2f}, min {dt_ms.min():.2f}, max {dt_ms.max():.2f},"
        f" median {median_ms:.2f}"
    )
    print(f"Slow frames : {int(np.sum(dt_ms > 1.5 * median_ms))} intervals > 1.5x median")
    print(f"Brightness  : mean pixel value {float(np.mean(brightness)):.1f} / 255 (measured frames)")
    return 0


def measure_with_config(cfg, seconds: float) -> int:
    """Measure through CameraCapture with the recorder's settings (timestamps = t_host_ns, after retrieve)."""
    from ekagrata.io.camera import CameraCapture, describe

    print(describe(cfg) + " (settings from configs/camera.yaml)")
    cap = CameraCapture(cfg)
    try:
        info = cap.start()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    try:
        print(f"Requested   : {info['requested']}, exposure_auto {cfg.exposure_auto}, "
              f"exposure {cfg.exposure}")
        print(f"Reported    : {info['reported']} (driver-reported, not measured)")
        for _ in range(WARMUP_FRAMES):
            cap.get(timeout_s=2.0)
        stamps_ns, brightness, shape = [], [], None
        deadline = time.perf_counter_ns() + int(seconds * 1e9)
        while time.perf_counter_ns() < deadline:
            frame = cap.get(timeout_s=1.0)
            if frame is None:
                if not cap.alive:
                    print(f"ERROR: {cap.error}")
                    return 1
                continue
            stamps_ns.append(frame.t_host_ns)
            brightness.append(float(frame.image.mean()))
            shape = frame.image.shape
        dropped = cap.dropped
    finally:
        cap.stop()
    code = report(stamps_ns, shape, brightness)
    print(f"Dropped     : {dropped} (capture queue full)")
    return code


def open_camera(cv2, index: int):
    for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
        backend = getattr(cv2, backend_name, None)
        if backend is None:
            continue
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap, backend_name
        cap.release()
    return None, None


def measure_raw(index: int, seconds: float, width: int, height: int, fps: int) -> int:
    try:
        import cv2
    except ImportError:
        print("cv2 not importable; skipping camera check.")
        return 1

    cap, backend_name = open_camera(cv2, index)
    if cap is None:
        print(f"No camera found at index {index} (tried CAP_DSHOW, CAP_MSMF).")
        return 1

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Some DirectShow drivers only accept the codec after the resolution is set; request it again.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        print(f"Camera: device index {index}, backend {backend_name} (--raw: no config, exposure untouched)")
        print(f"Requested   : {width}x{height} @ {fps} fps, MJPG")
        rep_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rep_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Reported    : {rep_w}x{rep_h} @ {cap.get(cv2.CAP_PROP_FPS):.2f} fps,"
            f" fourcc={fourcc_str!r} (driver-reported, not measured)"
        )
        if fourcc_str != "MJPG":
            print("WARNING: driver reports a different fourcc than requested: "
                  "probably not the intended camera.")

        # Warm-up: discard the first frames while exposure/auto-settings settle.
        for _ in range(WARMUP_FRAMES):
            cap.read()

        stamps_ns, brightness, shape = [], [], None
        deadline = time.perf_counter_ns() + int(seconds * 1e9)
        while time.perf_counter_ns() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            stamps_ns.append(time.perf_counter_ns())  # after read() = after retrieve (frame arrival + decode)
            brightness.append(float(frame.mean()))
            shape = frame.shape
    finally:
        cap.release()
    return report(stamps_ns, shape, brightness)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, help="camera index (default: `device` in configs/camera.yaml)")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--config", type=Path, default=Path("configs/camera.yaml"))
    parser.add_argument("--raw", action="store_true",
                        help="ignore camera.yaml backend/exposure (old behaviour)")
    parser.add_argument("--width", type=int, default=1280, help="--raw only")
    parser.add_argument("--height", type=int, default=720, help="--raw only")
    parser.add_argument("--fps", type=int, default=60, help="--raw only")
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be >= 0")
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    for name in ("width", "height", "fps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be > 0")

    print_versions()
    print()
    from ekagrata.core.config import ConfigError, load_camera_config

    try:
        cfg = load_camera_config(args.config)
    except ConfigError as exc:
        if not args.raw:
            print(f"ERROR: {exc}")
            return 2
        cfg = None
    device = args.camera if args.camera is not None else (cfg.device if cfg is not None else 0)
    if args.raw:
        return measure_raw(device, args.seconds, args.width, args.height, args.fps)
    return measure_with_config(dataclasses.replace(cfg, device=device), args.seconds)


if __name__ == "__main__":
    sys.exit(main())
