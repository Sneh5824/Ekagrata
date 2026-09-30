"""Zip the Blender add-on (legacy bl_info format) for Blender 5.0 "Install from Disk".

  uv run python scripts/package_blender_addon.py            # -> blender/ekagrata_live.zip
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADDON_DIR = REPO / "blender" / "addon" / "ekagrata_live"


def package(out: Path, addon_dir: Path = ADDON_DIR) -> list[str]:
    """Write `out` with the add-on under an `ekagrata_live/` folder (no __pycache__). Returns the archive
    names."""
    files = sorted(p for p in addon_dir.rglob("*.py") if "__pycache__" not in p.parts)
    if not (addon_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"{addon_dir / '__init__.py'} not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            name = f"{addon_dir.name}/{p.relative_to(addon_dir).as_posix()}"
            zf.write(p, name)
            names.append(name)
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=REPO / "blender" / "ekagrata_live.zip")
    args = p.parse_args()
    names = package(args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes): {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
