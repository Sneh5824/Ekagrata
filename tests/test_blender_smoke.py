"""Mandatory headless smoke test with the REAL Blender from configs/blender.yaml (SPEC M5-preview).

Skipped only if blender_exe does not exist (the reason says so). Builds the raw camera shadow scene, checks
its objects and the add-on inside Blender, and installs the packaged add-on zip into a temporary user folder.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from ekagrata.core.config import load_blender_config

REPO = Path(__file__).resolve().parents[1]
CFG = load_blender_config(REPO / "configs" / "blender.yaml")
EXE = Path(CFG.blender_exe)

pytestmark = pytest.mark.skipif(
    not EXE.is_file(), reason=f"blender_exe not found: {EXE} (smoke test is mandatory whenever it exists)")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cmd, env=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300, env=env)
    return r, f"exit {r.returncode}\n--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-4000:]}"


def test_blender_version_matches_config():
    expr = "import bpy, sys; print('EKAGRATA_VERSION', bpy.app.version_string, sys.version.split()[0])"
    r, log = run([EXE, "--background", "--factory-startup", "--python-expr", expr])
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("EKAGRATA_VERSION")), "")
    print(line)
    assert r.returncode == 0 and line.split()[1] == CFG.blender_version, log


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("blender") / "ekagrata_preview.blend"
    r, log = run(_load("build_blender_preview").build_command(CFG, out))
    assert r.returncode == 0 and out.is_file(), log
    return out


def test_scene_and_addon_smoke(scene):
    r, log = run([EXE, "--background", "--factory-startup", scene, "--python-exit-code", "1",
                  "--python", REPO / "blender" / "tests" / "smoke_check.py", "--",
                  "--offset", *(repr(v) for v in CFG.placement_offset_m), "--landmarks", *CFG.landmarks])
    print("\n".join(ln for ln in r.stdout.splitlines() if "ok:" in ln or "EKAGRATA" in ln))
    assert r.returncode == 0 and "EKAGRATA SMOKE OK" in r.stdout, log


def test_packaged_addon_installs_legacy(tmp_path):
    zip_path = tmp_path / "ekagrata_live.zip"
    names = _load("package_blender_addon").package(zip_path)
    assert names == ["ekagrata_live/__init__.py", "ekagrata_live/core.py"]
    env = dict(os.environ, BLENDER_USER_RESOURCES=str(tmp_path / "user"))  # never touch the real config
    r, log = run([EXE, "--background", "--factory-startup", "--python-exit-code", "1",
                  "--python", REPO / "blender" / "tests" / "install_check.py", "--", zip_path], env=env)
    print("\n".join(ln for ln in r.stdout.splitlines() if "EKAGRATA" in ln))
    assert r.returncode == 0 and "EKAGRATA INSTALL OK" in r.stdout, log
