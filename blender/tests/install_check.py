"""Install the packaged add-on zip the way 5.0's "Install from Disk" does for a legacy (bl_info) add-on, then
enable it. Run INSIDE Blender 5.0 with BLENDER_USER_* environment variables pointing at a temporary folder,
so the user's real Blender configuration is untouched:

  blender --background --factory-startup --python-exit-code 1 --python blender/tests/install_check.py -- <zip>

Blender 5.0.1 source (scripts/addons_core/bl_pkg): Install from Disk -> pkg_is_legacy_addon(zip) is True for a
zip without blender_manifest.toml whose .py files contain 'bl_info' -> exec_legacy() ->
PREFERENCES_OT_addon_install. This script calls that same legacy operator. Prints 'EKAGRATA INSTALL OK'.
"""

import sys
from pathlib import Path

import addon_utils
import bpy

zip_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
print(f"[EKAGRATA install] Blender {bpy.app.version_string}, user scripts: "
      f"{bpy.utils.user_resource('SCRIPTS')}")
result = bpy.ops.preferences.addon_install(filepath=str(zip_path), overwrite=True)
if result != {"FINISHED"}:
    raise AssertionError(f"addon_install returned {result}")
addon_utils.enable("ekagrata_live", default_set=True, handle_error=None)
if not hasattr(bpy.types, "EKAGRATA_PT_live") or not hasattr(bpy.ops.ekagrata, "live_start"):
    raise AssertionError("add-on installed but its panel/operators are not registered")
mod = sys.modules["ekagrata_live"]
installed = Path(mod.__file__).resolve()
if Path(bpy.utils.user_resource("SCRIPTS")).resolve() not in installed.parents:
    raise AssertionError(f"add-on loaded from {installed}, not from the (temporary) user scripts folder")
print(f"[EKAGRATA install] enabled from {installed}")
addon_utils.disable("ekagrata_live", default_set=True)
print("EKAGRATA INSTALL OK")
