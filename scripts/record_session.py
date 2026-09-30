"""Record a session: live camera or video file -> MediaPipe PoseLandmarker -> session folder.

Examples:
  uv run python scripts/record_session.py --athlete a01 --session live01 --preview --seconds 30
  (live camera = `device` in configs/camera.yaml; --camera N overrides it)
  uv run python scripts/record_session.py --athlete a01 --session phone01 --video D:/clips/smash.mp4
"""

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from ekagrata.core.config import MODEL_VARIANTS, ConfigError, load_camera_config
from ekagrata.core.timebase import now_ns
from ekagrata.io.camera import describe
from ekagrata.io.session import SessionWriter, base_metadata, create_session, sha256_file
from ekagrata.sources.pose import LiveCameraSource, VideoFileSource, pts_report
from ekagrata.vision.pose_landmarker import (
    create_landmarker,
    detect,
    draw_skeleton,
    has_pose,
    model_path,
    video_timestamp_ms,
)

WINDOW = "EKAGRATA recorder (q / Esc to stop)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--athlete", required=True, help="athlete id (letters, digits, '-')")
    p.add_argument("--session", required=True, help="session name (letters, digits, '-')")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int,
                     help="camera index (default: `device` in configs/camera.yaml); live is the default mode")
    src.add_argument("--video", type=Path, help="process a video file instead of the live camera")
    p.add_argument("--model", choices=MODEL_VARIANTS, help="PoseLandmarker variant (default: from config)")
    p.add_argument("--config", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--sessions-dir", type=Path, default=Path("data/sessions"))
    p.add_argument("--save-video", action="store_true",
                   help="live: write video/cam0.mp4; file: copy the source video into video/")
    p.add_argument("--preview", action="store_true", help="show a preview window with skeleton and stats")
    p.add_argument("--seconds", type=float, help="stop after this many seconds of footage")
    p.add_argument("--handedness", choices=("right", "left"), default="right")
    args = p.parse_args()
    if args.camera is not None and args.camera < 0:
        p.error("--camera must be >= 0")
    if args.seconds is not None and args.seconds <= 0:
        p.error("--seconds must be > 0")
    return args


def summarize(inference_ns: list[int], detected: int, processed: int) -> dict:
    inf_ms = np.asarray(inference_ns, dtype=np.float64) * 1e-6
    return {
        "frames_processed": processed,
        "inference_ms_mean": float(inf_ms.mean()) if inf_ms.size else None,
        "inference_ms_p95": float(np.percentile(inf_ms, 95)) if inf_ms.size else None,
        "detection_rate": detected / processed if processed else None,
    }


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def main() -> int:
    args = parse_args()
    try:
        cfg = load_camera_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.camera is not None:
        cfg = dataclasses.replace(cfg, device=args.camera)
    variant = args.model or cfg.model_variant
    mpath = model_path(args.models_dir, variant)
    if not mpath.is_file():
        print(f"ERROR: model file not found: {mpath}. Run: uv run python scripts/download_models.py")
        return 2
    live = args.video is None
    if live:
        print(describe(cfg))

    try:
        source = LiveCameraSource(cfg) if live else VideoFileSource(args.video)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2

    meta = base_metadata(args.session, args.athlete, args.handedness)
    info = source.info()
    # Live info carries timestamp_point "after_retrieve" (frame arrival + decode); file mode uses the
    # container PTS.
    meta["camera"] = {**info, "mode": "live" if live else "file"}
    meta["camera"].setdefault("timestamp_point", "pts")
    meta["model"] = {"name": mpath.name, "variant": variant, "file_sha256": sha256_file(mpath)}
    if not live:
        meta["camera"]["source_sha256"] = sha256_file(args.video)
    try:
        session_dir = create_session(args.sessions_dir, args.session, args.athlete, meta)
    except (FileExistsError, ValueError) as exc:
        source.close()
        print(f"ERROR: {exc}")
        return 2
    print(f"Session: {session_dir}")
    if not live and args.save_video:
        shutil.copy2(args.video, session_dir / "video" / f"cam0{args.video.suffix.lower()}")

    landmarker = create_landmarker(mpath)
    video_writer = None
    t_stamps: list[int] = []
    inference_ns: list[int] = []
    detected = 0
    last_ms = None
    t0 = None
    stop_reason = "end of input"

    try:
        with SessionWriter(session_dir) as writer:
            try:
                for frame in source.frames():
                    if t0 is None:
                        t0 = frame.t_host_ns
                    if args.seconds is not None and frame.t_host_ns - t0 >= args.seconds * 1e9:
                        stop_reason = "--seconds reached"
                        break
                    ts_ms = video_timestamp_ms(frame.t_host_ns, t0, last_ms)
                    last_ms = ts_ms
                    # M1: no clock sync yet; t_sync_ns = host clock (live) or PTS (file). See session.json.
                    t_sync_ns = frame.t_host_ns
                    t_a = now_ns()
                    pose = detect(landmarker, frame, ts_ms, t_sync_ns)
                    inference_ns.append(now_ns() - t_a)
                    detected += has_pose(pose)
                    writer.add(frame, pose)
                    t_stamps.append(frame.t_host_ns)

                    if live and args.save_video:
                        if video_writer is None:
                            # Container fps is nominal (requested); true frame times are in cam0_frames.csv.
                            h, w = frame.image.shape[:2]
                            video_writer = cv2.VideoWriter(
                                str(session_dir / "video" / "cam0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                float(cfg.fps), (w, h))
                        video_writer.write(frame.image)

                    if args.preview:
                        view = frame.image.copy()
                        draw_skeleton(view, pose)
                        recent = t_stamps[-30:]
                        fps = (len(recent) - 1) / ((recent[-1] - recent[0]) * 1e-9) if len(recent) > 1 and \
                            recent[-1] > recent[0] else None
                        drops = source.capture.dropped if live else 0
                        text = (f"fps {fmt(fps, '.1f')} | inference {inference_ns[-1] * 1e-6:.1f} ms | "
                                f"drops {drops} | pose {'yes' if has_pose(pose) else 'no'}")
                        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.imshow(WINDOW, view)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            stop_reason = "user stopped (q/Esc)"
                            break
            except KeyboardInterrupt:
                stop_reason = "Ctrl+C"

            summary = summarize(inference_ns, detected, len(t_stamps))
            summary["stop_reason"] = stop_reason
            if live:
                cap = source.capture
                summary.update({
                    "frames_captured": cap.captured,
                    "frames_dropped": cap.dropped,
                    "capture_fps_measured": cap.measured_fps(),
                })
                meta_camera = {**meta["camera"], "measured_fps": cap.measured_fps()}
            else:
                summary["pts"] = pts_report(t_stamps)
                meta_camera = {**meta["camera"], "measured_fps": summary["pts"].get("measured_fps")}
            if video_writer is not None:
                video_writer.release()
            writer.finalize({"summary": summary, "camera": meta_camera})
    finally:
        source.close()
        landmarker.close()
        if args.preview:
            cv2.destroyAllWindows()

    print("\n=== Measured summary ===")
    print(f"Stop reason       : {stop_reason}")
    if live:
        print(f"Frames captured   : {summary['frames_captured']}")
        print(f"Frames dropped    : {summary['frames_dropped']} (capture queue full)")
        print(f"Capture fps       : {fmt(summary['capture_fps_measured'])} (measured from frame timestamps)")
    else:
        pts = summary["pts"]
        print(f"Declared fps      : {fmt(info['reported']['fps'])} (container, not measured)")
        print(f"PTS fps           : {fmt(pts.get('measured_fps'))} (measured from timestamps)")
        print(f"PTS interval ms   : median {fmt(pts.get('interval_ms_median'), '.3f')}, "
              f"min {fmt(pts.get('interval_ms_min'), '.3f')}, max {fmt(pts.get('interval_ms_max'), '.3f')}")
        print(f"PTS irregular     : {pts.get('irregular', 'n/a')} intervals (>25% from median), "
              f"non-increasing {pts.get('non_increasing', 'n/a')}")
    print(f"Frames processed  : {summary['frames_processed']}")
    print(f"Inference ms      : mean {fmt(summary['inference_ms_mean'])}, "
          f"p95 {fmt(summary['inference_ms_p95'])} (model {variant})")
    rate = summary["detection_rate"]
    print(f"Detection rate    : {fmt(None if rate is None else 100 * rate, '.1f')} % of processed frames")
    print(f"Files             : {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
