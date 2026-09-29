# EKAGRATA

Athlete Digital Twin + Motion Shadow for the badminton forehand smash (Robofest Gujarat 6.0).

Pipeline (planned): camera + 3 wearable IMUs (BNO055 + ESP32, ESP-NOW → USB receiver) → sync → fusion →
joint angles / racket state → Digital Twin → smash simulation → ML → optimization.

Current state: **Milestone M0** — project foundation (quaternion math, time base, shared types, environment check).

## Setup (Windows 11, PowerShell)
Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```powershell
uv sync
```

## Run lint and tests
```powershell
uv run ruff check .
uv run pytest -q
```

## Check the environment and camera
Prints library versions, installed OpenCV distributions, and the camera frame rate **measured** from
`perf_counter_ns` timestamps (alongside what the driver reports).

```powershell
uv run python scripts/check_env.py --seconds 5
# options: --camera 0 --seconds 5 --width 1280 --height 720 --fps 60
```

## Record a session (M1)
```powershell
uv run python scripts/download_models.py                       # once: PoseLandmarker lite/full/heavy + SHA-256
uv run python scripts/record_session.py --athlete a01 --session live01 --camera 0 --preview --seconds 30
uv run python scripts/record_session.py --athlete a01 --session phone01 --video D:\clips\smash.mp4
uv run python scripts/benchmark_pose.py --video D:\clips\smash.mp4   # measured inference ms per variant
```
Camera settings live in `configs/camera.yaml`. Sessions are written to
`data/sessions/<YYYY-MM-DD>_<session>_<athlete>/` (never overwritten). Session and athlete names may contain
letters, digits and `-` only. Press `q`/Esc in the preview (or Ctrl+C) to stop; files are finalised either way.

## Joint angles (M2)
```powershell
uv run python scripts/check_axes.py                                        # live: confirm camera axis mapping
uv run python scripts/compute_joints.py --session data\sessions\<folder>   # -> motion\joints.parquet + plots
```
Stand far enough from the camera that **shoulders, hips and the right arm** are visible: shoulder angles need
the hips. Filter and derivative parameters are in `configs/joints.yaml`.

## Layout
```
ekagrata/core/     quat.py (only place for quaternion/SciPy conversion), timebase.py, types.py, config.py
ekagrata/io/       camera.py (capture thread), session.py (session folder write/read)
ekagrata/sources/  pose.py (LiveCameraSource, VideoFileSource)
ekagrata/vision/   pose_landmarker.py (MediaPipe Tasks API, VIDEO mode), landmark_map.py (MediaPipe -> world)
ekagrata/kinematics/ angles.py (segment frames, joint angles; camera and quaternion inputs)
ekagrata/analysis/ filters.py, derivatives.py
configs/ scripts/ tests/ docs/
data/sessions/     raw recordings (gitignored, append-only)
models/            model files (gitignored)
```

See [docs/conventions.md](docs/conventions.md) for units, quaternion, frame and time conventions.
