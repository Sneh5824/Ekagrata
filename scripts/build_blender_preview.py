"""Build the Blender RAW CAMERA SHADOW scene (M5-preview) with the pinned Blender from configs/blender.yaml.

Blender's own Python cannot read YAML, so this wrapper passes offset / landmarks / hitting side as arguments.
  uv run python scripts/build_blender_preview.py            # -> blender/ekagrata_preview.blend
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ekagrata.core.config import BlenderConfig, ConfigError, load_blender_config

REPO = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO / "blender" / "scripts" / "build_preview_scene.py"


def build_command(cfg: BlenderConfig, out: Path) -> list[str]:
    return [
        cfg.blender_exe, "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(BUILD_SCRIPT), "--",
        "--out", str(out),
        "--offset", *(repr(v) for v in cfg.placement_offset_m),
        "--hitting-side", cfg.hitting_side,
        "--landmarks", *cfg.landmarks,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO / "configs" / "blender.yaml")
    p.add_argument("--out", type=Path, default=REPO / "blender" / "ekagrata_preview.blend")
    args = p.parse_args()
    try:
        cfg = load_blender_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not Path(cfg.blender_exe).is_file():
        print(f"ERROR: blender_exe not found: {cfg.blender_exe} (edit {args.config})")
        return 2
    result = subprocess.run(build_command(cfg, args.out.resolve()), capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    for line in result.stdout.splitlines():
        if "EKAGRATA" in line or "Error" in line or "Traceback" in line:
            print(line)
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:], sep="\n", file=sys.stderr)
        print(f"ERROR: Blender exited with code {result.returncode}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
