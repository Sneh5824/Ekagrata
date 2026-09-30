"""Window layout arithmetic of scripts/live_shadow.py (no windows are opened)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("live_shadow", REPO / "scripts" / "live_shadow.py")
live_shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_shadow)


def test_split_screen_known_answer():
    # 1920x1080 at 125 % scaling -> logical 1536x864, taskbar 48 logical px -> work area 1536x816
    g = live_shadow.split_screen(0, 0, 1536, 816, 864, 1.25)
    fw, fh = live_shadow.CV_FRAME_W, live_shadow.CV_FRAME_H
    assert g["camera"] == (0, 0, 768 - fw, 816 - fh)  # OpenCV image area, logical px, top-left origin
    assert g["blender"] == (960, 60, 960, 1020)  # physical px; y = lower-left corner measured from the bottom
    at_100 = live_shadow.split_screen(0, 0, 1920, 1040, 1080)  # 100 % scaling
    assert at_100["blender"] == (960, 40, 960, 1040)
