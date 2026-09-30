"""Side-by-side live view: camera image (left) and the Blender RAW CAMERA SHADOW (right), one command.

  uv run python scripts/live_shadow.py                    # builds the scene if missing
  uv run python scripts/live_shadow.py --rebuild          # after changing configs/blender.yaml
  uv run python scripts/live_shadow.py --record swings04 --athlete 002   # also record a session

Launches Blender 5.0 (own window, own console; it stays open) with blender/ekagrata_preview.blend and
blender/scripts/live_autostart.py (starts the listener, looks through cam_matched), then runs
scripts/stream_to_blender.py --preview in this console. Ctrl+C here or q in the camera window stops the
streamer only. The raw camera shadow is NOT the Digital Twin; axis mapping unverified until check_axes passes.
"""

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

from ekagrata.core.config import ConfigError, load_blender_config, load_camera_config

REPO = Path(__file__).resolve().parents[1]
BLEND = REPO / "blender" / "ekagrata_preview.blend"
AUTOSTART = REPO / "blender" / "scripts" / "live_autostart.py"
POSTURE_NOTE = ("MediaPipe world coordinates are hip-centred; the figure shows posture only, not movement "
                "around the court (no footwork/translation).")


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)]


# Outer frame of an OpenCV window around its image area (Windows 11, measured 2026-09-30: 16 x 39 logical px).
CV_FRAME_W, CV_FRAME_H = 16, 39


def work_area() -> tuple[int, int, int, int, int, float]:
    """(left, top, right, bottom, screen_height, scale) of the primary monitor's work area (excludes the
    taskbar) in LOGICAL pixels, and the display scale (physical / logical). This process stays DPI-unaware
    like the OpenCV preview; Blender is DPI-aware and needs physical pixels. Windows only (ctypes)."""
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    rect = _Rect()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
    hdc = user32.GetDC(0)
    scale = gdi32.GetDeviceCaps(hdc, 118) / gdi32.GetDeviceCaps(hdc, 8)  # DESKTOPHORZRES / HORZRES
    user32.ReleaseDC(0, hdc)
    return rect.left, rect.top, rect.right, rect.bottom, user32.GetSystemMetrics(1), scale  # SM_CYSCREEN


def split_screen(left: int, top: int, right: int, bottom: int, screen_h: int, scale: float = 1.0) -> dict:
    """Left half for the camera window, right half for Blender.
    camera: (x, y, image w, image h) in logical pixels, top-left origin; the image area is shrunk by the
    window frame so the whole window fits. blender: --window-geometry (x, y, w, h) in PHYSICAL pixels with
    y = the window's LOWER-left corner measured from the bottom of the screen."""
    half = (right - left) // 2
    height = bottom - top
    camera = (left, top, half - CV_FRAME_W, height - CV_FRAME_H)
    blender = tuple(round(v * scale) for v in (left + half, screen_h - bottom, right - left - half, height))
    return {"camera": camera, "blender": blender}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rebuild", action="store_true", help="rebuild blender/ekagrata_preview.blend first")
    p.add_argument("--camera", type=int, help="camera index (default: `device` in configs/camera.yaml)")
    p.add_argument("--smooth", action="store_true", help="One-Euro smoothing in the streamer")
    p.add_argument("--seconds", type=float, help="stop the streamer after this many seconds")
    p.add_argument("--record", metavar="NAME", help="also record a session named NAME (needs --athlete)")
    p.add_argument("--athlete", help="athlete id for --record")
    p.add_argument("--no-arrange", action="store_true", help="do not place the windows side by side")
    args = p.parse_args()
    if args.record and not args.athlete:
        p.error("--record needs --athlete")
    return args


def main() -> int:
    args = parse_args()
    try:
        bcfg = load_blender_config(REPO / "configs" / "blender.yaml")
        load_camera_config(REPO / "configs" / "camera.yaml")
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not Path(bcfg.blender_exe).is_file():
        print(f"ERROR: blender_exe not found: {bcfg.blender_exe}")
        return 2

    newer = [c for c in ("blender.yaml", "camera.yaml")
             if BLEND.is_file() and (REPO / "configs" / c).stat().st_mtime > BLEND.stat().st_mtime]
    if args.rebuild or not BLEND.is_file():
        print("Building the scene ...")
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "build_blender_preview.py")])
        if r.returncode != 0:
            return r.returncode
    elif newer:
        print(f"WARNING: configs/{', configs/'.join(newer)} changed after the scene was built; camera "
              "placement may be out of date. Re-run with --rebuild.")

    geometry = None if args.no_arrange else split_screen(*work_area())
    blender_cmd = [bcfg.blender_exe, "--factory-startup"]
    if geometry:
        blender_cmd += ["--window-geometry", *(str(v) for v in geometry["blender"])]
    blender_cmd += [str(BLEND), "--python", str(AUTOSTART), "--", "--port", str(bcfg.udp_port)]
    # Own console: Ctrl+C here does not reach Blender, and Blender stays open afterwards.
    subprocess.Popen(blender_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=REPO)
    mc = bcfg.matched_camera
    print(f"Blender started: right half of the screen, viewing through cam_matched "
          f"({'matched view' if mc.yaw_deg == 0 else 'alternative viewpoint'}), listener on 127.0.0.1:"
          f"{bcfg.udp_port}. Its messages: Blender menu Window > Toggle System Console.")
    print("Camera window 'EKAGRATA camera': left half of the screen.")
    print("If the windows overlap, click one and press Win+Left, the other and press Win+Right.")
    print("NOTE: raw camera shadow, NOT the Digital Twin; axis mapping unverified until check_axes passes.")
    print(f"NOTE: {POSTURE_NOTE}")
    print("Stop the streamer with Ctrl+C here or q in the camera window; Blender stays open.\n", flush=True)

    stream_cmd = [sys.executable, str(REPO / "scripts" / "stream_to_blender.py"), "--preview"]
    if geometry:
        stream_cmd += ["--preview-geometry", *(str(v) for v in geometry["camera"])]
    if args.camera is not None:
        stream_cmd += ["--camera", str(args.camera)]
    if args.smooth:
        stream_cmd.append("--smooth")
    if args.seconds is not None:
        stream_cmd += ["--seconds", str(args.seconds)]
    if args.record:
        stream_cmd += ["--record", args.record, "--athlete", args.athlete]
    streamer = subprocess.Popen(stream_cmd, cwd=REPO)
    try:
        return streamer.wait()
    except KeyboardInterrupt:  # the streamer got the same Ctrl+C and prints its summary
        return streamer.wait()


if __name__ == "__main__":
    sys.exit(main())
