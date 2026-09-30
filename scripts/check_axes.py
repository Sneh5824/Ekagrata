"""Live check of the MediaPipe -> world axis mapping (configs/camera.yaml `mp_to_world`).

Stand FACING the camera, FULL BODY visible, head to feet. Follow the prompts: rest, right arm FORWARD, right
arm to the SIDE, right arm UP. For each movement the script prints how the right wrist moved in WORLD axes
(X forward/toward camera, Y left, Z up) and whether that matches the expected axis. Right after the rest step
the script aborts if either hip or the right wrist has median visibility < 0.5 in the rest samples (framing).
It also prints a camera pitch/roll ESTIMATE from the torso direction during the rest step (diagnostic only,
assumes the torso is vertical when standing relaxed; not applied anywhere).

Usage: uv run python scripts/check_axes.py [--camera N] [--model full] [--no-preview]
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

from ekagrata.core.config import MODEL_VARIANTS, ConfigError, load_camera_config
from ekagrata.io.camera import describe
from ekagrata.sources.pose import LiveCameraSource
from ekagrata.vision.camera_tilt import estimate_camera_tilt, torso_vector
from ekagrata.vision.framing import draw_framing, framing_status, rest_gate, wrist_edge
from ekagrata.vision.landmark_map import LM, mp_to_world
from ekagrata.vision.pose_landmarker import (
    create_landmarker,
    detect,
    draw_skeleton,
    model_path,
    video_timestamp_ms,
)

# (key, prompt, seconds). Samples are collected in the second half of each phase with a key != "".
PHASES = [
    ("", "Get ready: stand FACING the camera, FULL BODY visible, head to feet", 4.0),
    ("rest", "Arms relaxed at your sides - hold still", 4.0),
    ("forward", "Raise RIGHT arm straight FORWARD (toward camera), shoulder height - hold", 5.0),
    ("", "Lower the arm", 3.0),
    ("side", "Raise RIGHT arm straight out to YOUR RIGHT side, shoulder height - hold", 5.0),
    ("", "Lower the arm", 3.0),
    ("up", "Raise RIGHT arm straight UP overhead - hold", 5.0),
]
AXES = ("X (forward)", "Y (left)", "Z (up)")
REST_INDEX = next(i for i, (k, _, _) in enumerate(PHASES) if k == "rest")
GATE_LANDMARKS = ("l_hip", "r_hip", "r_wrist")


def verdict(key: str, d: np.ndarray) -> tuple[bool, str]:
    """Expected: forward -> +X dominates horizontally; side -> -Y (athlete's right) dominates horizontally;
    up -> +Z dominates overall."""
    if key == "forward":
        return bool(d[0] > 0 and abs(d[0]) > abs(d[1])), "+X should dominate X/Y"
    if key == "side":
        return bool(d[1] < 0 and abs(d[1]) > abs(d[0])), "-Y (your right) should dominate X/Y"
    return bool(d[2] > 0 and abs(d[2]) == np.max(np.abs(d))), "+Z should dominate"


def print_tilt_estimate(torso_vectors, cfg) -> None:
    """Diagnostic only: camera pitch/roll implied by the torso direction during the rest step."""
    pitch, roll, n = estimate_camera_tilt(torso_vectors)
    print(f"\nCamera tilt ESTIMATE from the torso during the rest step (n = {n} frames; diagnostic only, not "
          "applied):")
    if n == 0:
        print("  no rest frames with world landmarks: cannot estimate")
        return
    print(f"  pitch {pitch:+.1f} deg (positive = camera looks down), roll {roll:+.1f} deg (positive = camera "
          "rotated clockwise, seen from behind the camera)")
    print("  ASSUMPTION: torso vertical when standing relaxed; any lean adds directly to these numbers.")
    print("  EKAGRATA 'Z up' currently means camera-up. Measure the real tilt with a spirit-level app and")
    print("  record it as camera_pitch_deg / camera_roll_deg in configs/camera.yaml "
          f"(now {cfg.camera_pitch_deg:+.1f} / {cfg.camera_roll_deg:+.1f}; not yet applied).\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, help="camera index (default: configs/camera.yaml)")
    p.add_argument("--model", choices=MODEL_VARIANTS, help="PoseLandmarker variant (default: from config)")
    p.add_argument("--config", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()
    if args.camera is not None and args.camera < 0:
        p.error("--camera must be >= 0")
    try:
        cfg = load_camera_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.camera is not None:
        cfg = dataclasses.replace(cfg, device=args.camera)
    print(describe(cfg))
    mpath = model_path(args.models_dir, args.model or cfg.model_variant)
    try:
        landmarker = create_landmarker(mpath)
        source = LiveCameraSource(cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    bounds = np.cumsum([0.0] + [s for _, _, s in PHASES])
    samples: dict[str, list[np.ndarray]] = {k: [] for k, _, _ in PHASES if k}
    rest_vis: dict[str, list[float]] = {n: [] for n in GATE_LANDMARKS}
    rest_torso: list[np.ndarray] = []
    t0, last_ms, shown, gate_done = None, None, -1, False
    try:
        for frame in source.frames():
            t0 = frame.t_host_ns if t0 is None else t0
            el = (frame.t_host_ns - t0) * 1e-9
            i = int(np.searchsorted(bounds, el, side="right")) - 1
            if i >= len(PHASES):
                break
            key, prompt, dur = PHASES[i]
            if not gate_done and i > REST_INDEX:
                gate_done = True
                failed = rest_gate(rest_vis)
                if failed:
                    print("\nABORTED after the rest step: framing problem (median visibility over the "
                          "rest samples, threshold 0.5):")
                    for name, med in failed:
                        print(f"  {name:<8}: median visibility {med:.2f}")
                    print("Step back or lower the camera until you are visible head to feet "
                          "(hips and hands in frame), then run check_axes again.")
                    return 1
                print_tilt_estimate(rest_torso, cfg)
            if i != shown:
                print(f"[{i + 1}/{len(PHASES)}] {prompt}")
                shown = i
            last_ms = video_timestamp_ms(frame.t_host_ns, t0, last_ms)
            pose = detect(landmarker, frame, last_ms, frame.t_host_ns)
            wrist_vis = pose.landmarks_img[LM["r_wrist"], 3]
            if key == "rest" and el >= bounds[i] + dur / 2:  # NaN (no pose) counts as 0 in rest_gate
                for n in GATE_LANDMARKS:
                    rest_vis[n].append(float(pose.landmarks_img[LM[n], 3]))
                if pose.landmarks_world is not None:
                    rest_torso.append(torso_vector(mp_to_world(pose.landmarks_world, cfg.mp_to_world)))
            if key and el >= bounds[i] + dur / 2 and pose.landmarks_world is not None and wrist_vis >= 0.5:
                samples[key].append(mp_to_world(pose.landmarks_world[LM["r_wrist"]], cfg.mp_to_world))
            if not args.no_preview:
                view = frame.image.copy()
                draw_skeleton(view, pose)
                draw_framing(view, framing_status(pose.landmarks_img), wrist_edge(pose.landmarks_img))
                cv2.putText(view, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(view, f"{bounds[i + 1] - el:4.1f} s", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                cv2.imshow("EKAGRATA check_axes (q / Esc to abort)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Aborted by user.")
                    return 1
    except KeyboardInterrupt:
        print("Aborted (Ctrl+C).")
        return 1
    finally:
        source.close()
        landmarker.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print("\nmp_to_world used:", [list(r) for r in cfg.mp_to_world])
    if len(samples["rest"]) < 3:
        print(f"ERROR: only {len(samples['rest'])} rest samples with a visible right wrist; cannot evaluate.")
        return 1
    rest = np.median(np.array(samples["rest"]), axis=0)
    all_ok = True
    print("Right-wrist displacement from rest, WORLD frame [m] (median of the hold window):")
    for key in ("forward", "side", "up"):
        if len(samples[key]) < 3:
            print(f"  {key:<8}: too few samples ({len(samples[key])}) with a visible right wrist "
                  "-> NOT EVALUATED")
            all_ok = False
            continue
        d = np.median(np.array(samples[key]), axis=0) - rest
        ok, rule = verdict(key, d)
        all_ok &= ok
        dom = AXES[int(np.argmax(np.abs(d)))]
        print(f"  {key:<8}: dX={d[0]:+.3f} dY={d[1]:+.3f} dZ={d[2]:+.3f}  (n={len(samples[key])}, "
              f"largest: {dom})  {'OK' if ok else 'MISMATCH'} - {rule}")
    print("\nRESULT:", "mapping CONFIRMED for this camera setup" if all_ok else
          "mapping NOT confirmed - paste this output; mp_to_world needs to change")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
