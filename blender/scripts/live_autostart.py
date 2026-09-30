"""Auto-start the EKAGRATA live link after the file loads (side-by-side live view; used by
scripts/live_shadow.py).

  blender --factory-startup blender/ekagrata_preview.blend
      --python blender/scripts/live_autostart.py -- --port 9870

Blender 5.0, bpy + standard library only. In a UI session a bpy.app.timers callback waits for a window,
registers the add-on from blender/addon if it is not registered yet, starts the listener, and makes the 3D
view look through the scene camera (cam_matched) with reduced overlays. In background mode (-b) nothing is
registered.
"""

import argparse
import sys
from pathlib import Path

import bpy

ADDON_DIR = Path(__file__).resolve().parents[1] / "addon"
RETRY_S = 0.5
_settings = {"port": 9870}


def _ensure_addon():
    """Return the ekagrata_live module, registering it from blender/addon if no copy is registered yet."""
    if str(ADDON_DIR) not in sys.path:
        sys.path.insert(0, str(ADDON_DIR))
    import ekagrata_live

    if not hasattr(bpy.types, "EKAGRATA_PT_live"):
        ekagrata_live.register()
    return ekagrata_live


def _view3d(window):
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                return area, region
    return None, None


def _setup_view(space) -> None:
    """Look through the scene camera; hide grid/axes/cursor/relationship lines/origins/text. Extras stay on
    (they draw the landmark Empties)."""
    space.region_3d.view_perspective = "CAMERA"
    overlay = space.overlay
    overlay.show_floor = False
    overlay.show_axis_x = False
    overlay.show_axis_y = False
    overlay.show_cursor = False
    overlay.show_relationship_lines = False
    overlay.show_object_origins = False
    overlay.show_text = False
    space.shading.type = "SOLID"
    space.shading.color_type = "MATERIAL"


def _start():
    """Timer callback: retry until a window with a 3D view exists, then start the listener once."""
    wm = bpy.context.window_manager
    if wm is None or not wm.windows:
        return RETRY_S
    window = wm.windows[0]
    area, region = _view3d(window)
    if area is None:
        return RETRY_S
    mod = _ensure_addon()
    core = mod.core
    wm.ekagrata_port = _settings["port"]
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.ekagrata.live_start()
        _setup_view(area.spaces.active)
        bpy.ops.view3d.view_center_camera()
    print(f"[EKAGRATA autostart] add-on from {mod.__file__}; live_start -> {result}; "
          f"port {_settings['port']}")
    cam = bpy.context.scene.camera
    print(f"[EKAGRATA autostart] view: through {cam.name} ({cam.get('ekagrata_view', 'unknown view')}); "
          "raw camera shadow, NOT the Digital Twin")
    print(f"[EKAGRATA autostart] {core.AXIS_CAVEAT}")
    print(f"[EKAGRATA autostart] {core.POSTURE_NOTE}")
    return None


def install(background: bool) -> bool:
    """Register the auto-start timer unless Blender runs in background mode. Returns True if registered."""
    if background:
        print("[EKAGRATA autostart] background mode: live auto-start skipped")
        return False
    if not bpy.app.timers.is_registered(_start):
        bpy.app.timers.register(_start, first_interval=1.0)
    return True


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="EKAGRATA live auto-start")
    p.add_argument("--port", type=int, default=9870)
    _settings["port"] = p.parse_args(argv).port
    install(bpy.app.background)


if __name__ == "__main__":
    main()
