"""Download MediaPipe PoseLandmarker model files into models/ and record SHA-256 in models/manifest.json.

Usage: uv run python scripts/download_models.py [--variants lite full heavy] [--models-dir models] [--force]
"""

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ekagrata.io.session import sha256_file

URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/"
       "pose_landmarker_{v}.task")
VARIANTS = ("lite", "full", "heavy")


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    failed = 0
    for v in args.variants:
        url = URL.format(v=v)
        dest = args.models_dir / f"pose_landmarker_{v}.task"
        if dest.is_file() and not args.force:
            print(f"{dest.name}: already present (use --force to re-download)")
        else:
            print(f"{dest.name}: downloading {url}")
            try:
                download(url, dest)
            except (urllib.error.URLError, OSError) as exc:
                print(f"{dest.name}: DOWNLOAD FAILED: {exc}")
                failed += 1
                continue
        digest = sha256_file(dest)
        manifest[dest.name] = {
            "url": url,
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "recorded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        print(f"{dest.name}: {dest.stat().st_size} bytes, sha256 {digest}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
