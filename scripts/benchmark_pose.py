"""Benchmark PoseLandmarker variants on the same video file: measured inference time and detection rate.

Usage: uv run python scripts/benchmark_pose.py --video PATH [--variants lite full heavy] [--max-frames N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ekagrata.core.config import MODEL_VARIANTS
from ekagrata.core.timebase import now_ns
from ekagrata.sources.pose import VideoFileSource
from ekagrata.vision.pose_landmarker import (
    create_landmarker,
    detect,
    has_pose,
    model_path,
    video_timestamp_ms,
)


def run_variant(video: Path, mpath: Path, max_frames: int | None) -> dict:
    source = VideoFileSource(video)
    landmarker = create_landmarker(mpath)
    inference_ns, detected, last_ms, t0 = [], 0, None, None
    try:
        for frame in source.frames():
            if max_frames is not None and len(inference_ns) >= max_frames:
                break
            t0 = frame.t_host_ns if t0 is None else t0
            last_ms = video_timestamp_ms(frame.t_host_ns, t0, last_ms)
            t_a = now_ns()
            pose = detect(landmarker, frame, last_ms, frame.t_host_ns)
            inference_ns.append(now_ns() - t_a)
            detected += has_pose(pose)
    finally:
        landmarker.close()
        source.close()
    ms = np.asarray(inference_ns, dtype=np.float64) * 1e-6
    n = len(ms)
    return {
        "frames": n,
        "mean_ms": float(ms.mean()) if n else float("nan"),
        "p95_ms": float(np.percentile(ms, 95)) if n else float("nan"),
        "detection_pct": 100.0 * detected / n if n else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=MODEL_VARIANTS, default=list(MODEL_VARIANTS))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--max-frames", type=int, help="process at most N frames per variant")
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be > 0")
    if not args.video.is_file():
        print(f"ERROR: video file not found: {args.video}")
        return 2
    missing = [v for v in args.variants if not model_path(args.models_dir, v).is_file()]
    if missing:
        print(f"ERROR: model file(s) missing for {missing}. Run: uv run python scripts/download_models.py")
        return 2

    print(f"Video: {args.video}  (measured on this machine, MediaPipe default CPU delegate)")
    print(f"{'variant':<8} {'frames':>7} {'mean ms':>9} {'p95 ms':>9} {'detected %':>11}")
    for v in args.variants:
        r = run_variant(args.video, model_path(args.models_dir, v), args.max_frames)
        print(f"{v:<8} {r['frames']:>7} {r['mean_ms']:>9.2f} {r['p95_ms']:>9.2f} {r['detection_pct']:>11.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
