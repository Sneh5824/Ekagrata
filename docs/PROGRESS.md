# EKAGRATA — Progress Log

The agent updates this file at the end of every milestone. The user commits after checking the evidence.

## Status
| Milestone | Status | Date | Commit |
|---|---|---|---|
| M0 Foundation | DONE | 2026-09-29 | (not yet committed — no git repo) |
| M1 Camera + MediaPipe recorder | DONE (phone-video step waived) | 2026-09-29 | (not yet committed — no git repo) |
| M2 Joint angles + filtering | IN PROGRESS | 2026-09-29 | |
| M3 Segmentation, features, labels, shuttle speed | TODO | | |
| M4 Kinematic model, FK, IK, generators | TODO | | |
| M5 Blender twin + UDP live link | TODO | | |
| M6 Protocol, IMU sources, IMU simulator | TODO | | |
| M7 Sync, calibration, fusion (simulated) | TODO | | |
| M7b EKF fusion | TODO | | |
| M8 Smash sim + optimization | TODO | | |
| M8b MuJoCo inverse dynamics | TODO | | |
| M9 Live pipeline + latency | TODO | | |
| M10 ROS 2 (spike, then adapters) | TODO | | |
| M11 Firmware | TODO | | |
| M12 ML pipeline | TODO | | |
| M13 RL (gated) | TODO | | |
| M14 Experiments + evidence pack | TODO | | |

Status values: TODO · IN PROGRESS · BLOCKED (reason) · DONE

## Milestone reports
<!-- Template for each milestone:
### Mx — name — YYYY-MM-DD
- Files changed:
- Tests: N passed (paste the pytest summary line)
- Ruff: (paste output)
- Acceptance evidence: one line per criterion + command + real output excerpt
- [USER] steps performed and their output:
- Known limitations:
- Suggested commit message:
-->

### M0 — Foundation — 2026-09-29
- Files changed:
  - `pyproject.toml` (Python `==3.12.*`, deps per SPEC M0, dev: pytest, ruff; ruff line length 110; pytest
    testpaths=tests), `uv.lock`, `.python-version`, `.gitignore`, `README.md`
  - `ekagrata/__init__.py`, `ekagrata/core/{__init__,quat,timebase,types}.py`, `ekagrata/io/__init__.py`,
    `ekagrata/vision/__init__.py`
  - `scripts/check_env.py`
  - `tests/test_quat.py`, `tests/test_timebase.py`
  - `docs/conventions.md`, `docs/PROGRESS.md`
  - `configs/.gitkeep`, `data/sessions/.gitkeep`, `models/.gitkeep`
- Tests: `39 passed in 0.42s`
- Ruff: `All checks passed!`
- Acceptance evidence:
  - Fresh `uv sync` works — repo copied without `.venv`/`data`/`models` to a temp dir; `uv sync --locked` installed
    all packages; in that copy `uv run ruff check .` → `All checks passed!`, `uv run pytest -q` → `39 passed in 1.41s`.
  - Exactly one OpenCV distribution — `check_env.py` → `OpenCV dists: ['opencv-contrib-python==5.0.0.93']`
    (pulled by mediapipe; no second OpenCV added).
  - ruff + pytest pass — outputs above.
  - `check_env.py` prints measured fps — `uv run python scripts/check_env.py --seconds 5` (run by the agent on
    this laptop, camera index 0):
    ```
    Backend     : CAP_DSHOW
    Requested   : 1280x720 @ 60 fps, MJPG
    Reported    : 1280x720 @ 60.00 fps, fourcc='MJPG' (driver-reported, not measured)
    Frame shape : (720, 1280, 3)
    Measured    : 15.02 fps over 4.99 s (76 frames)
    Interval ms : mean 66.57, min 60.90, max 80.93, median 64.05
    Slow frames : 0 intervals > 1.5x median
    ```
  - Script usability (§6) — `--help` prints usage; `--seconds 0` → `check_env.py: error: --seconds must be > 0`
    (exit code 2).
  - SPEC M0 test list — covered in `tests/test_quat.py` / `tests/test_timebase.py`: SciPy round-trips with
    explicit ordering assertion; 90° about Z maps (1,0,0)→(0,1,0); composition law; `relative` identity and 30°;
    `angle_between(q,-q)=0`; slerp endpoints/midpoint/shortest arc; seeded random comparisons vs SciPy (multiply,
    rotate_vector, rotvec, slerp); swing–twist recomposition (random q and axes), pure twist, pure swing, signed
    angle, 180° range, singular 180°-perpendicular case; `unwrap_u32` across one and two wraps.
- [USER] steps performed and their output: user ran `uv run python scripts/check_env.py --seconds 5`:
  ```
  Backend     : CAP_DSHOW
  Requested   : 1280x720 @ 60 fps, MJPG
  Reported    : 1280x720 @ 60.00 fps, fourcc='MJPG' (driver-reported, not measured)
  Frame shape : (720, 1280, 3)
  Measured    : 15.02 fps over 4.99 s (76 frames)
  Interval ms : mean 66.56, min 62.54, max 80.81, median 64.24
  Slow frames : 0 intervals > 1.5x median
  ```
  Follow-up by the agent (same camera): `--width 640 --height 480` → `Measured : 15.02 fps over 4.99 s (76 frames)`;
  `--width 640 --height 480 --fps 30` → `Measured : 15.02 fps over 4.99 s (76 frames)`.
- Known limitations:
  - The driver reports 60 fps but the measured rate is 15.02 fps, identical at 1280×720@60, 640×480@60 and
    640×480@30 (runs above). Because it does not change with resolution or requested fps, bandwidth is ruled out;
    the cap comes from the camera itself (auto-exposure lengthening the frame time is the leading hypothesis —
    NOT verified). Must be resolved or accepted before M1 recordings.
  - Before the MJPG request was repeated after setting the resolution, the camera delivered YUY2 and measured
    9.98 fps (earlier run, same laptop).
  - `PoseFrame` does not yet carry `pts_ns` / `dropped_before` (§4.1 `cam0_frames.csv`); deferred to M1 where that
    file is written.
  - Swing–twist convention chosen: `axis` in the child frame, `q = swing ⊗ twist`, twist canonicalised to w ≥ 0,
    angle in (−π, π]; identity twist when undefined. Documented in `docs/conventions.md`.
- Suggested commit message:
  ```
  M0: project foundation (uv project, quaternion/time core, types, env check, conventions)
  ```

### M1 — Camera capture + MediaPipe recorder — 2026-09-29 (DONE; phone-video step waived by the user)
- Plan decisions (approved defaults): optional `exposure_auto`/`exposure` in `camera.yaml` (null = untouched);
  `t_sync_ns = t_host_ns` (live) or PTS (file) until M7, recorded in `session.json.clock_sync.method`; video PTS
  via OpenCV `CAP_PROP_POS_MSEC` (no PyAV); `camera.yaml` holds only M1 keys, unknown keys rejected
  (`mp_to_world` arrives in M2, LED ROI in M7); hard kill loses the Parquet footer (Ctrl+C/exceptions are safe);
  `--video` records source path + SHA-256, `--save-video` copies the file.
- Files changed:
  - new: `ekagrata/core/config.py`, `configs/camera.yaml`, `ekagrata/io/camera.py`, `ekagrata/io/session.py`,
    `ekagrata/sources/__init__.py`, `ekagrata/sources/pose.py`, `ekagrata/vision/pose_landmarker.py`,
    `scripts/download_models.py`, `scripts/record_session.py`, `scripts/benchmark_pose.py`,
    `tests/test_config.py`, `tests/test_camera.py`, `tests/test_sources_pose.py`, `tests/test_session.py`,
    `tests/test_pose_landmarker.py`
  - modified: `ekagrata/core/types.py` (new `CameraFrame`), `README.md`, `docs/PROGRESS.md`
  - downloaded (gitignored): `models/pose_landmarker_{lite,full,heavy}.task`, `models/manifest.json`
- Tests: `79 passed in 6.91s` (with model files present; without them the landmarker test is skipped)
- Ruff: `All checks passed!`
- Acceptance evidence (agent smoke runs; sessions written to a temp dir, not `data/sessions/`):
  - Models + SHA-256 — `uv run python scripts/download_models.py`:
    lite 5777746 bytes sha256 `59929e1d…574a`; full 9398198 bytes `4eaa5eb7…c4ad`;
    heavy 30664242 bytes `64437af8…bc7b` (full hashes in `models/manifest.json`).
  - Video-file path — synthetic 60-frame 30 fps clip, `record_session.py --video … --model lite`:
    `PTS fps : 30.00 (measured from timestamps)`, `PTS interval ms : median 33.333, min 33.333, max 33.333`,
    `PTS irregular : 0`, `Frames processed : 60`, `Inference ms : mean 10.40, p95 10.92 (model lite)`,
    `Detection rate : 0.0 %` (no person in the clip — expected).
  - Live path — `record_session.py --camera 0 --model full --seconds 5 --save-video`:
    `Frames captured : 75`, `Frames dropped : 0`, `Capture fps : 14.78 (measured from grab timestamps)`,
    `Frames processed : 74`, `Inference ms : mean 17.95, p95 27.82 (model full)`, `Detection rate : 41.9 %`.
  - Parquet loads with the documented columns — `read_session(...)` on the live smoke session:
    `parquet columns == documented: True | n_cols 266 | rows 74`; frames CSV columns
    `['frame_idx', 't_host_ns', 't_sync_ns', 'pts_ns', 'dropped_before']`; `t_sync strictly increasing: True`;
    `session.json` contains every §4.3 key plus `summary`, `n_frames`, `finalized_utc`.
  - SPEC M1 tests: session write→read round-trip (incl. multiple Parquet row groups, NaN/None for no-pose frames,
    exception mid-write still finalises, never overwrites); timestamps strictly increasing (capture thread with a
    fake camera, `video_timestamp_ms`, file PTS); VideoFileSource on a synthetic `cv2.VideoWriter` clip;
    PoseLandmarker wrapper test (skipped when the model file is absent).
- [USER] steps performed and their output:
  1. DONE — 30 s live recording, `uv run python scripts/record_session.py --athlete 002 --session live01
     --camera 0 --preview --seconds 30`:
     ```
     Stop reason       : --seconds reached
     Frames captured   : 451
     Frames dropped    : 0 (capture queue full)
     Capture fps       : 14.98 (measured from grab timestamps)
     Frames processed  : 450
     Inference ms      : mean 15.86, p95 16.56 (model full)
     Detection rate    : 100.0 % of processed frames
     ```
     Agent check with `read_session` on `2026-09-29_live01_002` (and the earlier `2026-09-29_live01_001`): columns
     match `landmark_columns()`, 450 landmark rows = 450 frame rows, `t_sync_ns` strictly increasing, span
     29.98 s, `dropped_before` sum 0, `session.json` finalised, not aborted.
  2. WAIVED by the user (2026-09-29: "skip this phone part, only do on webcam laptop") — SPEC M1 acceptance
     "processes one phone video". The video-file code path is verified only on the synthetic clip above; it has
     NOT been exercised on a real phone file (240 fps PTS spacing, VFR handling are unverified).
  3. Optional `benchmark_pose.py` — not run (needs a video file).
- Known limitations:
  - Laptop webcam capture is ~15 fps measured (M0 and the smoke run above). The driver reports
    `auto_exposure: -1.0` (probably "not supported/unknown" on this driver) and `exposure: -6.0`
    (≈ 1/64 s in the DSHOW log2 convention), which is shorter than the measured frame interval — so the
    auto-exposure hypothesis is weakened but NOT verified either way.
  - MediaPipe logs `Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI` for the
    non-square live frames. Its effect on landmark accuracy is not verified.
  - `frames captured` (75) can exceed `frames processed` (74): frames grabbed after the stop condition, or still
    queued, are counted as captured but not processed.
  - Saved live video (`cam0.mp4`) declares the requested fps in its container; the true frame times are only in
    `cam0_frames.csv`.
  - `code_version` is `"unknown (no git)"` until git is initialised.
- Suggested commit message (after the [USER] steps pass):
  ```
  M1: camera capture, MediaPipe PoseLandmarker recorder, session files, model download and benchmark
  ```

### M2 — Landmark mapping, joint angles, filtering, derivatives — 2026-09-29 (IN PROGRESS: waiting for [USER] steps)
- Plan decisions (approved defaults): facing-camera `mp_to_world` confirmed by `check_axes.py`; camera wrist
  angles assume zero pronation (quality "low"), `racket_twist` NaN; parameters in new `configs/joints.yaml`;
  landmark positions are filtered (not angles); left-handed sessions exit with an error; elbow/wrist joint
  quaternions NaN for camera; Butterworth cutoff validated < fs/2; `compute_joints` appends to
  `session.json.derived[]` (raw CSV/Parquet untouched).
- Files changed:
  - new: `ekagrata/vision/landmark_map.py`, `ekagrata/kinematics/{__init__,angles}.py`,
    `ekagrata/analysis/{__init__,filters,derivatives}.py`, `configs/joints.yaml`, `scripts/check_axes.py`,
    `scripts/compute_joints.py`, `tests/test_landmark_map.py`, `tests/test_angles.py`, `tests/test_filters.py`,
    `tests/test_derivatives.py`, `tests/test_compute_joints.py`
  - modified: `ekagrata/core/quat.py` (+`from_matrix`, `to_matrix`, `from_two_vectors`), `ekagrata/core/config.py`
    (+`mp_to_world`, `JointsConfig`), `configs/camera.yaml` (+`mp_to_world`), `tests/test_quat.py`,
    `tests/test_config.py`, `tests/test_camera.py`, `docs/conventions.md`, `README.md`, `docs/PROGRESS.md`
- Tests: `133 passed` · Ruff: `All checks passed!`
- SPEC M2 tests → where: synthetic skeletons (arm down elev 0; horizontal abduction elev π/2 plane 0; forward
  flexion plane π/2; elbow straight 0 / 90° bend π/2) → `test_angles.py`; invariance to a global rotation of all
  landmarks → `test_global_rotation_invariance`; derivative of a known sinusoid → `test_savgol_derivative_of_sinusoid`
  (tolerance 1 % of peak for the 1st, 5 % for the 2nd derivative — SG attenuation of the 2nd derivative measured
  at 3.2 % in that test); NaN propagation → `test_missing_landmark_propagates_nan`, filter/derivative gap tests;
  filtfilt phase lag ≈ 0 → `test_filtfilt_zero_phase_lag_and_passband` (cross-correlation peak at lag 0).
  Extra: ISB Y-X-Y recovery, camera path == quaternion path on 20 random poses, PDF example 135° → flex 45°,
  mirror `mp_to_world` rejected, SIMULATED end-to-end pipeline test (`test_compute_joints.py`).
- Agent smoke run of `compute_joints.py` on a COPY of `2026-09-29_live01_002` (temp dir; real session untouched):
  `Frame rate : 15.60 Hz (measured, median interval)`; coverage elbow_flex 75.0 %, wrist_flex/dev 60.9 %,
  shoulder angles 0.0 %. Cause: hips never visible in that recording (median visibility l_hip 0.00, r_hip 0.00),
  so the torso frame is undefined. The script now prints this diagnosis.
- [USER] steps still required (SPEC M2 acceptance):
  1. `uv run python scripts/check_axes.py` — stand facing the laptop with hips visible; paste the output.
  2. Record ~10 smash-like swings with shoulders, HIPS and right arm in view:
     `uv run python scripts/record_session.py --athlete 002 --session swings01 --preview --seconds 40`.
  3. `uv run python scripts/compute_joints.py --session data\sessions\<date>_swings01_002` → paste the output
     and check `motion\plots\joint_angles.png` / `joint_velocities.png`.
- Known limitations:
  - Camera at ~15 fps (measured): Butterworth cutoff must be < 7.5 Hz and the SG window (7 samples) spans
    ~0.45 s, so camera angular velocities are heavily smoothed; they cannot resolve the smash acceleration
    phase (SPEC §1.4). Plots state the measured fs.
  - Camera wrist angles assume zero pronation (unobservable) → quality "low".
  - `shoulder_plane` is still computed near elev 0, where it is ill-defined (SPEC only requires `shoulder_rot`
    to be NaN there).
  - Left-handed mirroring (SPEC §3.1) not implemented; such sessions are refused with a clear error.
- Suggested commit message (after the [USER] steps pass):
  ```
  M2: landmark->world mapping, 3D joint angles (camera + quaternion paths), filters, derivatives, compute_joints
  ```

### Camera diagnosis before M2 data collection — 2026-09-29 (user-requested task, not a SPEC milestone)
- Files: new `scripts/camera_probe.py`, `tests/test_camera_probe.py`; modified `configs/camera.yaml`
  (`exposure_auto: false`, `exposure: -6.0`), `README.md`, `docs/PROGRESS.md`.
- `uv run python scripts/camera_probe.py --camera 0`: 32 combinations (dshow/msmf × 640x480/1280x720 ×
  MJPG/default × auto/-5/-6/-7), each 1.5 s warm-up + 3 s measurement; run twice with consistent results.
  Second run (excerpt; fps MEASURED from perf_counter_ns, dups = frames byte-identical to the previous one):
  ```
  dshow 1280x720 MJPG    auto -> 14.96 fps, median 64.2 ms, bright 116.7, dups 0
  dshow 1280x720 MJPG      -5 -> 29.90 fps, median 32.1 ms, bright  96.8, dups 0
  dshow 1280x720 MJPG      -6 -> 29.92 fps, median 32.0 ms, bright  58.0, dups 0
  dshow 1280x720 MJPG      -7 -> 29.91 fps, median 32.1 ms, bright  31.0, dups 0
  dshow 1280x720 default(YUY2) any exposure -> 9.96-9.98 fps
  msmf  any size/codec   auto -> 29.76-29.84 fps but median 53.7-56.6 ms and 45 duplicate frames
  msmf  any size/codec   manual -> 30.04-30.11 fps, dups 0, but driver always reports exposure -6.0
  ```
- Diagnosis (from the measurements): the 15 fps cap is auto-exposure — with a manual exposure the same camera
  delivers ~30 fps of distinct frames; brightness roughly halves per exposure step, so the driver applies it.
  MSMF + auto pads its output with duplicate frames (only ~half are new images), which a pure fps count would
  hide. Uncompressed 1280x720 over DSHOW is limited to ~10 fps.
- Chosen (probe rule: fastest usable, then larger resolution, exposure read back correctly, shorter exposure):
  dshow, 1280x720, MJPG, `exposure_auto: false`, `exposure: -6.0` (DSHOW log2 s → 1/64 s).
- Verification through the real recorder with the new config (temp dir):
  `Frames captured : 148`, `Frames dropped : 0`, `Capture fps : 29.37 (measured from grab timestamps)`,
  `Detection rate : 100.0 %`; session.json reported `exposure: -6.0`, `fourcc: 'MJPG'`.
- Tests: `137 passed`; Ruff: `All checks passed!`
- Limitations: brightness 58.0/255 was measured in the lighting at probe time; darker recording conditions may
  need exposure -5 (re-run the probe). A 1/64 s exposure still blurs a fast racket/wrist. The probe re-enables
  auto-exposure at the end, so `check_env.py` (which does not set exposure) will again show ~15 fps; the
  recorder applies `camera.yaml` exposure itself.
- Found while verifying (NOT changed, needs your decision): `record_session.py` requires `--camera` or `--video`,
  so the `device` in `camera.yaml` is never used as a default. Earlier advice that commands without `--camera`
  use the config was wrong; always pass `--camera 0` for now.

## Open questions for the user
- `record_session.py` / `check_axes.py`: make `--camera` optional (default = `camera.yaml` `device`)? Currently
  `record_session.py` requires `--camera` or `--video`.
- Git is not initialised in this folder. Should the agent run `git init` (no commit), or will you?
- 2026-09-29: phone camera tested — Realme 7 via Iriun Webcam over USB is camera index 1. Per the user, the project
  uses the laptop webcam only for now, so `configs/camera.yaml` stays at `device: 0`. User run `check_env.py --camera 1`: `Measured : 30.01 fps over 5.00 s (151 frames)`,
  `Interval ms : mean 33.32, min 26.98, max 40.85, median 33.52`; driver-reported fourcc is not a valid code
  (`'}ë6ä'`, Iriun virtual driver). Resolution/fps are set in the Iriun apps; requested values may be ignored.
- RESOLVED 2026-09-29: laptop webcam ~15 fps was caused by auto-exposure; manual exposure -6 gives ~30 fps
  measured (see "Camera diagnosis" above).
