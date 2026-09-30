"""Probe a camera: MEASURE real fps and brightness for each backend / resolution / codec / exposure setting.

Nothing is assumed about what the camera supports: every combination is opened, requested, and measured from
perf_counter_ns frame timestamps. Driver-REPORTED values are printed separately from MEASURED ones.

Usage: uv run python scripts/camera_probe.py [--camera 0] [--seconds 3] [--warmup 1.5] [--min-brightness 40]
"""

import argparse
import itertools
import sys
import time

import cv2
import numpy as np

from ekagrata.io.camera import fourcc_to_str

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
SIZES = [(640, 480), (1280, 720)]
CODECS = ["MJPG", None]  # None = do not request a codec (driver default)
EXPOSURES = ["auto", -5.0, -6.0, -7.0]  # manual values in the DSHOW log2-seconds convention
AUTO_ON, AUTO_OFF = 0.75, 0.25  # DSHOW convention for CAP_PROP_AUTO_EXPOSURE


def probe(index: int, backend: str, size, codec, exposure, fps_req: int, warmup_s: float,
          seconds: float) -> dict:
    row = {"backend": backend, "w": size[0], "h": size[1], "codec": codec or "default", "exposure": exposure,
           "fps_req": fps_req}
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    if not cap.isOpened():
        cap.release()
        return {**row, "status": "open failed"}
    try:
        accepted = []
        if codec:
            accepted.append(cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec)))
        accepted.append(cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0]))
        accepted.append(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1]))
        accepted.append(cap.set(cv2.CAP_PROP_FPS, fps_req))
        if codec:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        if exposure == "auto":
            exp_ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_ON)
        else:
            exp_ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_OFF)
            exp_ok = cap.set(cv2.CAP_PROP_EXPOSURE, exposure) and exp_ok
        row.update({
            "set_ok": all(accepted), "exp_set_ok": bool(exp_ok),
            "rep_w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "rep_h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "rep_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "rep_fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "rep_auto": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "rep_exp": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
        })
        t_end = time.perf_counter_ns() + int(warmup_s * 1e9)
        while time.perf_counter_ns() < t_end:  # let exposure / format settle; frames discarded
            cap.read()
        stamps, bright, failures, shape, dups, prev = [], [], 0, None, 0, None
        t_end = time.perf_counter_ns() + int(seconds * 1e9)
        while time.perf_counter_ns() < t_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                continue
            stamps.append(time.perf_counter_ns())
            shape = frame.shape
            if prev is not None and np.array_equal(frame, prev):
                dups += 1  # byte-identical to the previous frame: not a new image
            prev = frame
            if len(stamps) % 3 == 1:
                bright.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    finally:
        cap.release()
    if len(stamps) < 3:
        return {**row, "status": f"too few frames ({len(stamps)}), read failures {failures}"}
    t = np.asarray(stamps, dtype=np.int64)
    dt_ms = np.diff(t) * 1e-6
    return {**row, "status": "ok", "frames": len(t), "failures": failures, "dups": dups,
            "shape": f"{shape[1]}x{shape[0]}", "fps_meas": (len(t) - 1) / ((t[-1] - t[0]) * 1e-9),
            "dt_med_ms": float(np.median(dt_ms)), "dt_max_ms": float(dt_ms.max()),
            "brightness": float(np.mean(bright))}


def restore_auto_exposure(index: int) -> None:
    for api in BACKENDS.values():
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_ON)
        cap.release()


def choose_best(rows: list[dict], min_brightness: float, fps_tol: float = 0.5) -> dict | None:
    """Usable = delivered frames at the requested size, mean brightness >= min_brightness, no duplicate
    frames, and median interval consistent with the measured rate (no bursty delivery). Among usable
    combinations within fps_tol of the highest MEASURED fps, prefer: larger resolution; a driver that reports
    the requested exposure back (truthful metadata); then the shorter exposure (less motion blur)."""
    ok = [r for r in rows if r.get("status") == "ok" and r["shape"] == f"{r['w']}x{r['h']}"
          and r["brightness"] >= min_brightness and r["dups"] == 0
          and abs(1000.0 / r["dt_med_ms"] - r["fps_meas"]) <= 0.1 * r["fps_meas"]]
    if not ok:
        return None
    top = max(r["fps_meas"] for r in ok)
    near = [r for r in ok if r["fps_meas"] >= top - fps_tol]

    def key(r):
        manual = r["exposure"] != "auto"
        readback = manual and abs(r["rep_exp"] - r["exposure"]) < 1e-6
        return (r["w"] * r["h"], readback, -r["exposure"] if manual else -np.inf)

    return max(near, key=key)


def fmt_row(r: dict) -> str:
    exp = r["exposure"] if r["exposure"] == "auto" else f"{r['exposure']:+.0f}"
    head = f"{r['backend']:<5} {r['w']:>4}x{r['h']:<4} {r['codec']:<7} {exp:>5} |"
    if r.get("status") != "ok":
        return f"{head} {r.get('status')}"
    return (f"{head} {r['rep_w']:>4}x{r['rep_h']:<4} {r['rep_fps']:6.2f} {r['rep_fourcc']!r:<8} "
            f"{r['rep_auto']:5.2f} {r['rep_exp']:5.1f} {'y' if r['exp_set_ok'] else 'n':>3} | "
            f"{r['shape']:>9} {r['fps_meas']:6.2f} {r['dt_med_ms']:6.1f} {r['dt_max_ms']:6.1f} "
            f"{r['brightness']:6.1f} {r['failures']:>4} {r['dups']:>4}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=int, default=60, help="fps to request (the driver may ignore it)")
    p.add_argument("--seconds", type=float, default=3.0, help="measurement window per combination")
    p.add_argument("--warmup", type=float, default=1.5, help="seconds discarded after applying settings")
    p.add_argument("--min-brightness", type=float, default=40.0,
                   help="mean gray level (0-255) below which a combination is considered too dark to use")
    args = p.parse_args()
    for name in ("seconds", "warmup"):
        if getattr(args, name) <= 0:
            p.error(f"--{name} must be > 0")
    if args.camera < 0 or args.fps <= 0:
        p.error("--camera must be >= 0 and --fps > 0")

    combos = list(itertools.product(BACKENDS, SIZES, CODECS, EXPOSURES))
    print(f"Camera {args.camera}: {len(combos)} combinations x ({args.warmup}s warm-up + {args.seconds}s "
          f"measurement). Keep the scene and lighting constant.\n")
    print(f"Camera: device index {args.camera}, backends dshow and msmf (sweep; the two backends may number "
          "devices differently)")
    print("REQUESTED                         | DRIVER-REPORTED (not measured)                  | MEASURED")
    print("back  size      codec     exp |  size      fps   fourcc   auto   exp  set |     shape    fps  "
          "med_ms max_ms bright fail dups")
    rows = []
    try:
        for backend, size, codec, exposure in combos:
            r = probe(args.camera, backend, size, codec, exposure, args.fps, args.warmup, args.seconds)
            rows.append(r)
            print(fmt_row(r), flush=True)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        restore_auto_exposure(args.camera)
        print("\n(auto-exposure re-enabled on both backends)")

    best = choose_best(rows, args.min_brightness)
    print("\n'set' = the driver accepted the exposure request (API return value, not proof it was applied).")
    print(f"'bright' = mean gray level 0-255 of the measured frames; usable threshold --min-brightness "
          f"{args.min_brightness}.")
    print("'dups' = frames byte-identical to the previous one. A median interval far from 1000/fps means "
          "bursty delivery; both make a combination unusable for timing.")
    if best is None:
        print("\nNo combination delivered frames at the requested size with sufficient brightness.")
        return 1
    exp = best["exposure"]
    print("\nHighest measured fps among usable combinations:")
    print(f"  {fmt_row(best)}")
    print("Suggested configs/camera.yaml values:")
    print(f"  backend: {best['backend']}\n  width: {best['w']}\n  height: {best['h']}")
    print(f"  fourcc: {best['codec'] if best['codec'] != 'default' else 'null  # do not request a codec'}")
    print("  exposure_auto: " + ("true" if exp == "auto" else "false"))
    print("  exposure: " + ("null" if exp == "auto" else f"{exp:.1f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
