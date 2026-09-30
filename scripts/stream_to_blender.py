"""Stream MediaPipe landmarks to the Blender RAW CAMERA SHADOW over UDP (M5-preview; NOT the Digital Twin).

Live (M1 capture + PoseLandmarker path; nothing is recorded; camera = `device` in configs/camera.yaml):
  uv run python scripts/stream_to_blender.py            # or --camera N to override the index
Replay a recorded session in real time (from t_sync_ns):
  uv run python scripts/stream_to_blender.py --session data/sessions/<dir> --loop
Add --smooth for the online One-Euro filter (parameters from configs/joints.yaml). Stop with Ctrl+C.
Live mode prints a per-stage latency breakdown on exit (--timing-csv PATH also writes the per-frame stamps);
method in docs/blender.md.
Points are mapped with camera.yaml mp_to_world: axis mapping unverified until check_axes passes.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ekagrata.analysis.filters import OneEuroFilter
from ekagrata.core.config import ConfigError, load_blender_config, load_camera_config, load_joints_config
from ekagrata.core.timebase import now_ns
from ekagrata.io.session import LANDMARKS_PARQUET
from ekagrata.transport.udp_twin import UdpTwinSender, world_points
from ekagrata.vision.landmark_map import N_LANDMARKS

AXIS_CAVEAT = "axis mapping unverified until check_axes passes"
COARSE_SLEEP_MARGIN_NS = 2_000_000  # sleep until ~1 ms before the target, then yield-spin to it

# Live latency stamps (host perf_counter_ns, one row per SENT message) and the stages derived from them.
# t_host_ns = t_sync_ns = after retrieve() (frame arrival + decode); t_grab_ns = after grab() (diagnostic).
STAMPS = ("t_grab_ns", "t_host_ns", "t_dequeue_ns", "t_infer_start_ns", "t_infer_end_ns", "t_send_ns",
          "t_sent_ns")
STAGES = {  # name: (from stamp, to stamp)
    "grab -> retrieve end": ("t_grab_ns", "t_host_ns"),
    "queue wait": ("t_host_ns", "t_dequeue_ns"),
    "dequeue -> inference": ("t_dequeue_ns", "t_infer_start_ns"),
    "inference": ("t_infer_start_ns", "t_infer_end_ns"),
    "map + smooth": ("t_infer_end_ns", "t_send_ns"),
    "encode + sendto": ("t_send_ns", "t_sent_ns"),
    "t_host -> sent": ("t_host_ns", "t_sent_ns"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int,
                     help="live camera index (default: `device` in configs/camera.yaml); "
                          "live is the default mode")
    src.add_argument("--session", type=Path, help="session folder to replay (pose/cam0_landmarks.parquet)")
    p.add_argument("--loop", action="store_true", help="replay: start again at the end")
    p.add_argument("--smooth", action="store_true", help="apply the online One-Euro filter")
    p.add_argument("--seconds", type=float, help="stop after this many seconds")
    p.add_argument("--config", type=Path, default=Path("configs/blender.yaml"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--joints-config", type=Path, default=Path("configs/joints.yaml"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--model", choices=("lite", "full", "heavy"), help="PoseLandmarker variant (live)")
    p.add_argument("--timing-csv", type=Path, help="live: write per-frame latency stamps to this CSV")
    args = p.parse_args(argv)
    if args.camera is not None and args.camera < 0:
        p.error("--camera must be >= 0")
    if args.seconds is not None and args.seconds <= 0:
        p.error("--seconds must be > 0")
    if args.loop and args.session is None:
        p.error("--loop only applies to --session")
    return args


def load_replay(session_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_sync_ns (N,), MediaPipe world landmarks (N, 33, 3), visibility (N, 33)) from a session folder."""
    path = Path(session_dir) / LANDMARKS_PARQUET
    if not path.is_file():
        raise FileNotFoundError(f"landmarks file not found: {path}")
    lms = pd.read_parquet(path)
    t = lms["t_sync_ns"].to_numpy(np.int64)
    if t.size == 0:
        raise ValueError(f"{path}: no frames")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: t_sync_ns is not strictly increasing")
    world = np.stack([lms[[f"wlm{i}_{k}" for k in "xyz"]].to_numpy(float) for i in range(N_LANDMARKS)],
                     axis=1)
    vis = np.stack([lms[f"lm{i}_vis"].to_numpy(float) for i in range(N_LANDMARKS)], axis=1)
    return t, world, vis


def wait_until(target_ns: int, clock, sleep) -> None:
    """Return once clock() >= target_ns (never early): coarse sleep, then yield until the target."""
    while True:
        remaining = target_ns - clock()
        if remaining <= 0:
            return
        sleep((remaining - COARSE_SLEEP_MARGIN_NS // 2) * 1e-9 if remaining > COARSE_SLEEP_MARGIN_NS else 0)


def run_replay(t_sync_ns, emit, clock=time.perf_counter_ns, sleep=time.sleep, loop=False,
               should_stop=lambda: False, max_passes=None) -> list[int]:
    """Call emit(i) for frame i at start + (t_sync_ns[i] - t_sync_ns[0]) on `clock`. With `loop`, the next
    pass starts one median frame interval after the last frame. Returns the lateness (ns) of every emit."""
    t = np.asarray(t_sync_ns, dtype=np.int64)
    rel = t - t[0]
    gap = int(np.median(np.diff(t))) if t.size > 1 else 0
    start = clock()
    lateness: list[int] = []
    passes = 0
    while True:
        for i, r in enumerate(rel):
            target = start + int(r)
            wait_until(target, clock, sleep)
            if should_stop():
                return lateness
            lateness.append(clock() - target)
            emit(i)
        passes += 1
        if not loop or (max_passes is not None and passes >= max_passes):
            return lateness
        start = start + int(rel[-1]) + gap


def stage_durations_ms(rows: list[dict]) -> dict:
    """{stage: durations in ms (array)} from per-message stamp rows (see STAGES)."""
    out = {}
    for name, (a, b) in STAGES.items():
        out[name] = np.array([(r[b] - r[a]) * 1e-6 for r in rows], dtype=np.float64)
    return out


def stage_report(rows: list[dict]) -> list[str]:
    """Median / p95 / max per stage, plus frame interval and queue wait expressed in frame intervals."""
    if not rows:
        return ["(no messages sent: no latency rows)"]
    lines = [f"{'stage':24s} {'median ms':>10s} {'p95 ms':>9s} {'max ms':>9s}   "
             f"(n = {len(rows)} sent messages)"]
    durations = stage_durations_ms(rows)
    for name, d in durations.items():
        lines.append(f"{name:24s} {np.median(d):10.2f} {np.percentile(d, 95):9.2f} {d.max():9.2f}")
    t_host = np.array([r["t_host_ns"] for r in rows], dtype=np.int64)
    if t_host.size > 1:
        interval = float(np.median(np.diff(t_host))) * 1e-6
        q = durations["queue wait"]
        lines.append(f"{'frame interval (sent)':24s} {interval:10.2f}   "
                     "(median between consecutive sent frames)")
        lines.append(f"queue wait / frame interval: median {np.median(q) / interval:.2f}, "
                     f"p95 {np.percentile(q, 95) / interval:.2f}")
    return lines


def make_smoother(enabled: bool, joints_config: Path):
    """Return f(t_ns, points: dict) -> dict, the One-Euro filter on world points (or identity)."""
    if not enabled:
        return lambda t_ns, pts: pts
    jc = load_joints_config(joints_config)
    filters: dict = {}

    def smooth(t_ns, pts):
        out = {}
        for name, p in pts.items():
            f = filters.setdefault(name, OneEuroFilter(jc.one_euro_min_cutoff_hz, jc.one_euro_beta,
                                                       jc.one_euro_d_cutoff_hz))
            out[name] = f(t_ns, np.asarray(p, dtype=np.float64)).tolist()
        return out

    return smooth


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def stream_replay(args, bcfg, cam, sender, smooth) -> dict:
    t, world, vis = load_replay(args.session)
    has_pose = np.isfinite(world[:, 0, 0])
    stats = {"frames_processed": 0, "frames_with_pose": 0}
    t_start = time.perf_counter_ns()

    def emit(i):
        pts, vs = world_points(world[i] if has_pose[i] else None, vis[i], bcfg.landmarks, cam.mp_to_world)
        sender.send_landmarks("replay", int(t[i]), smooth(int(t[i]), pts), vs)
        stats["frames_processed"] += 1
        stats["frames_with_pose"] += int(has_pose[i])

    def should_stop():
        return args.seconds is not None and time.perf_counter_ns() - t_start >= args.seconds * 1e9

    print(f"Replaying {t.size} frames ({(t[-1] - t[0]) * 1e-9:.2f} s){' in a loop' if args.loop else ''}")
    try:
        late = run_replay(t, emit, loop=args.loop, should_stop=should_stop)
        stats["stop_reason"] = "--seconds reached" if should_stop() else "end of session"
    except KeyboardInterrupt:
        late, stats["stop_reason"] = [], "Ctrl+C"
    if late:
        ms = np.asarray(late, dtype=np.float64) * 1e-6
        stats["lateness_ms"] = (float(np.median(ms)), float(np.percentile(ms, 95)), float(ms.max()))
    return stats


def stream_live(args, bcfg, cam, sender, smooth) -> dict:
    # Imported here so replay does not load OpenCV / MediaPipe.
    from ekagrata.io.camera import describe
    from ekagrata.sources.pose import LiveCameraSource
    from ekagrata.vision.pose_landmarker import (
        create_landmarker,
        detect,
        has_pose,
        model_path,
        video_timestamp_ms,
    )

    if args.camera is not None:
        cam = dataclasses.replace(cam, device=args.camera)
    print(describe(cam))
    mpath = model_path(args.models_dir, args.model or cam.model_variant)
    landmarker = create_landmarker(mpath)
    try:
        source = LiveCameraSource(cam)
    except RuntimeError:
        landmarker.close()
        raise
    info = source.info()
    print(f"        requested {info['requested']}, timestamp point {info['timestamp_point']}")
    print(f"        driver-reported (not measured) {info['reported']}")
    stats = {"frames_processed": 0, "frames_with_pose": 0, "stop_reason": "end of input", "timing": []}
    t0, last_ms = None, None
    try:
        for frame in source.frames():
            t_dequeue = now_ns()
            t0 = frame.t_host_ns if t0 is None else t0
            if args.seconds is not None and frame.t_host_ns - t0 >= args.seconds * 1e9:
                stats["stop_reason"] = "--seconds reached"
                break
            last_ms = video_timestamp_ms(frame.t_host_ns, t0, last_ms)
            # As in M1: no clock sync yet, t_sync_ns = host QPC clock at grab (Blender shows capture age).
            t_infer_start = now_ns()
            pose = detect(landmarker, frame, last_ms, frame.t_host_ns)  # incl. BGR->RGB + mp.Image
            t_infer_end = now_ns()
            ok = has_pose(pose)
            pts, vs = world_points(pose.landmarks_world if ok else None, pose.landmarks_img[:, 3],
                                   bcfg.landmarks, cam.mp_to_world)
            if sender.send_landmarks("real", pose.t_sync_ns, smooth(pose.t_sync_ns, pts), vs):
                stats["timing"].append(dict(zip(STAMPS, (
                    frame.t_grab_ns, frame.t_host_ns, t_dequeue, t_infer_start, t_infer_end,
                    sender.last_send_ns, sender.last_sent_ns), strict=True)) | {
                    "frame_idx": frame.frame_idx, "dropped_before": frame.dropped_before, "pose": int(ok)})
            stats["frames_processed"] += 1
            stats["frames_with_pose"] += int(ok)
    except KeyboardInterrupt:
        stats["stop_reason"] = "Ctrl+C"
    finally:
        stats["capture_fps"] = source.capture.measured_fps()
        stats["frames_dropped"] = source.capture.dropped
        source.close()
        landmarker.close()
    return stats


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bcfg = load_blender_config(args.config)
        cam = load_camera_config(args.camera_config)
        smooth = make_smoother(args.smooth, args.joints_config)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"EKAGRATA raw camera shadow -> udp://{bcfg.udp_host}:{bcfg.udp_port} "
          f"(max {bcfg.send_rate_max_hz:g} Hz, smoothing {'One-Euro' if args.smooth else 'off'})")
    print(f"NOTE: {AXIS_CAVEAT}")
    sender = UdpTwinSender(bcfg.udp_host, bcfg.udp_port, bcfg.send_rate_max_hz)
    try:
        if args.session is not None:
            stats = stream_replay(args, bcfg, cam, sender, smooth)
        else:
            stats = stream_live(args, bcfg, cam, sender, smooth)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        sender.close()

    n, n_pose = stats["frames_processed"], stats["frames_with_pose"]
    print("\n=== Measured summary ===")
    print(f"Stop reason       : {stats['stop_reason']}")
    print(f"Frames processed  : {n}")
    print(f"Frames with pose  : {n_pose} ({fmt(100.0 * n_pose / n if n else None, '.1f')} %)")
    print(f"Messages sent     : {sender.sent}")
    print(f"Rate-limited      : {sender.rate_limited} (skipped, max {bcfg.send_rate_max_hz:g} Hz)")
    last_error = f" ({sender.last_error})" if sender.last_error else ""
    print(f"Send errors       : {sender.send_errors}{last_error}")
    print(f"Send rate         : {fmt(sender.measured_rate_hz())} Hz (measured, first to last send)")
    if "lateness_ms" in stats:
        med, p95, mx = stats["lateness_ms"]
        print(f"Replay lateness   : median {med:.3f} ms, p95 {p95:.3f} ms, max {mx:.3f} ms (vs schedule)")
    if "capture_fps" in stats:
        print(f"Capture fps       : {fmt(stats['capture_fps'])} (measured from frame timestamps)")
        print(f"Frames dropped    : {stats['frames_dropped']} (capture queue full)")
    if "timing" in stats:
        print("\n=== Latency breakdown (host perf_counter_ns, measured; t_host = after retrieve) ===")
        print("\n".join(stage_report(stats["timing"])))
        print("Blender adds send -> apply: see the add-on's 'age since send' (panel / console on Stop).")
        if args.timing_csv is not None and stats["timing"]:
            args.timing_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(stats["timing"]).to_csv(args.timing_csv, index=False, encoding="utf-8")
            print(f"Per-frame stamps written: {args.timing_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
