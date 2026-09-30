"""Headless check of blender/scripts/live_autostart.py, run INSIDE Blender 5.0 in background mode.

  blender --background --factory-startup --python-exit-code 1 --python blender/tests/autostart_check.py

The auto-start timer must NOT be registered in background mode; install(background=False) must register it
(unregistered again here, since a background session has no window to start in).
Prints 'EKAGRATA AUTOSTART OK'.
"""

import importlib.util
from pathlib import Path

import bpy

path = Path(__file__).resolve().parents[1] / "scripts" / "live_autostart.py"
spec = importlib.util.spec_from_file_location("live_autostart", path)
autostart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autostart)  # module name is not "__main__", so main() does not run

if not bpy.app.background:
    raise AssertionError("this check must run with --background")
if autostart.install(bpy.app.background) or bpy.app.timers.is_registered(autostart._start):
    raise AssertionError("auto-start timer registered in background mode")
print("  ok: background mode -> no auto-start timer")
if not autostart.install(False) or not bpy.app.timers.is_registered(autostart._start):
    raise AssertionError("install(background=False) did not register the auto-start timer")
bpy.app.timers.unregister(autostart._start)
print("  ok: UI mode -> auto-start timer registered (then unregistered)")
print("EKAGRATA AUTOSTART OK")
