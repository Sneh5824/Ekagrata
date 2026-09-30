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
| M5-preview Blender raw camera shadow (out-of-order, user-approved) | IN PROGRESS (built; waiting for [USER] acceptance) | 2026-09-30 | |
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

### M5-preview — Blender raw camera shadow — 2026-09-30 (IN PROGRESS: built, waiting for [USER] acceptance)
Out-of-order milestone approved by the user on 2026-09-30. M2 stays IN PROGRESS; M3/M4/M5 stay TODO. Raw camera
shadow only: Empties at raw landmark positions; no `rig.json`, no joint rotations, no fixed segment lengths.
- Blender check (first step after approval), real binary from `configs/blender.yaml`:
  `blender --background --python-expr "import bpy, sys; print(bpy.app.version_string, sys.version)"` →
  `5.0.1 3.11.13 (main, Sep 23 2025, 09:08:45) [MSC v.1929 64 bit (AMD64)]`; `--version` → `Blender 5.0.1`,
  build hash `a3db93c5b259`, branch `blender-v5.0-release`.
- Add-on format: **legacy `bl_info`**. Evidence from Blender 5.0.1's own source
  (`5.0/scripts/addons_core/bl_pkg`): Install from Disk (`extensions.package_install_files`) calls
  `pkg_is_legacy_addon()` (zip without `blender_manifest.toml` whose `.py` contains `bl_info`) → `exec_legacy()`
  → `preferences.addon_install`. Verified by installing the packaged zip in the real 5.0.1 (test below).
  Note: `addon_utils.py` in 5.0.1 says `bl_info` will be fully deprecated later; switching means adding a manifest.
- bpy calls verified against docs.blender.org/api/5.0 and/or the running 5.0.1: see the approved plan.
  Found: `Material.use_nodes` is deprecated in 5.0 (no effect), so it is not used; `bpy.data.materials.new`
  creates a Principled BSDF node tree in 5.0.1 (checked at runtime).
- Files changed:
  - new: `configs/blender.yaml`, `ekagrata/transport/{__init__,udp_twin}.py`, `scripts/stream_to_blender.py`,
    `scripts/build_blender_preview.py`, `scripts/package_blender_addon.py`,
    `blender/scripts/build_preview_scene.py`, `blender/addon/ekagrata_live/{__init__,core}.py`,
    `blender/tests/{smoke_check,install_check}.py`, `docs/blender.md`, `tests/test_udp_twin.py`,
    `tests/test_stream_replay.py`, `tests/test_blender_live_core.py`, `tests/test_blender_smoke.py`
  - modified: `ekagrata/core/config.py` (+`BlenderConfig`, `load_blender_config`), `tests/test_config.py`,
    `pyproject.toml` (ruff `per-file-target-version` py311 for `blender/**`; pytest marker `timing`),
    `.gitignore` (+`blender/*.blend`, `*.blend1`, `blender/ekagrata_live.zip`), `docs/SPEC.md` (M5 Blender pin,
    new M5-preview section), `docs/PROGRESS.md`
  - generated (gitignored): `blender/ekagrata_preview.blend`, `blender/ekagrata_live.zip`
- Tests: `207 passed in 50.16s` (was 137; +70) · Ruff: `All checks passed!`
- Tests → requirement:
  - message round-trip (0.1 mm rounding), NaN left out + strict JSON, invalid messages rejected →
    `test_udp_twin.py`; sender survives no listener (200 sends) and never raises on `OSError`,
    `BlockingIOError`, `ConnectionResetError` → `test_sender_survives_no_listener`,
    `test_send_failure_never_raises`.
  - replay timing: fake-clock scheduler (authoritative) → `test_scheduler_fake_clock_exact`,
    `test_scheduler_loop_and_stop`; real clock, marked `@pytest.mark.timing` (runs by default; tolerance: never
    early by > 1 ms, median lateness ≤ 5 ms, max ≤ 30 ms, span within ±30 ms; failure message prints the
    measured numbers). Measured: `emit-time lateness min 0.124 ms; scheduler lateness median 0.002, p95 0.003,
    max 0.005; span error -0.003 ms`. Synthetic (SIMULATED) parquet end-to-end through a real loopback socket →
    `test_main_replay_end_to_end`.
  - keep-newest logic (pure Python, factored out of bpy) incl. duplicates/stale, seq gaps, sender restart,
    2 s windows, whole-run summary, replay capture age = n/a, `link_transform` known answers →
    `test_blender_live_core.py`.
  - mandatory headless smoke test with the real `blender_exe` → `test_blender_smoke.py` (3 tests):
    `EKAGRATA_VERSION 5.0.1 3.11.13`; `ok: all 42 expected objects exist`, court 13.40 m / 5.18 m, net 1.524 m,
    hitting-arm colours, add-on registered, every Empty at offset + point (1e-6 m), links span their landmarks
    (1e-5 m), vis < 0.5 hidden and re-shown, real-socket drain `3 valid + 1 bad`, `only the newest applied`,
    `EKAGRATA SMOKE OK`; packaged zip installed via the legacy operator into a temporary
    `BLENDER_USER_RESOURCES` → `EKAGRATA INSTALL OK`.
- Agent end-to-end runs in the Blender 5.0.1 GUI (real modal TIMER operator, started from a scratch script;
  not part of the test suite):
  - Replay of the real session `2026-09-29_swings01_002` (read only), `--loop --seconds 12`: sender
    `Messages sent : 358`, `Send rate : 29.83 Hz`, `Replay lateness : median 0.006 ms, p95 0.009 ms,
    max 0.033 ms`, `Rate-limited : 1` (two recorded frames closer than 1/60 s). Blender: `received 225,
    applied 225, superseded 0, stale 0, lost (seq gaps) 0, bad 0`; `last 2 s: rx 30.0 Hz`; `age since send:
    median 8.5 / p95 15.7 ms`; capture age `n/a in replay`; nose and l_shoulder moved off the placeholder;
    r_wrist and l_hip hidden (in that recording r_wrist has vis ≥ 0.5 in only 19.2 % of frames, hips in 0 %).
    Socket closed on Stop.
  - Live `--camera 0 --seconds 12` — **CORRECTION (2026-09-30 investigation): DirectShow index 0 was the Iriun
    virtual camera's 'waiting for phone' screen, not the laptop webcam; the earlier note 'nobody in front of
    the camera' was wrong, and these numbers are not representative**: `Frames processed : 99`,
    `Frames with pose : 0`, `Capture fps : 8.19 (measured from grab timestamps)`; Blender: `received 57,
    applied 57, lost 0`; `age since send: median 7.0 / p95 14.7 ms`; **`age since capture: median 144.5 /
    p95 152.0 ms`** (capture → Blender apply, excluding display; includes MediaPipe inference). All landmarks
    hidden (no pose).
- Acceptance (SPEC M5-preview):
  - [x] Blender 5.0.1 pinned, bpy + stdlib only, headless smoke test with the real binary passes (above).
  - [x] Panel shows measured receive rate and message age (send + capture, median/p95 over 2 s); printed on Stop.
  - [ ] [USER] replayed session visibly drives the raw camera shadow in Blender 5.0.1.
  - [ ] [USER] live camera visibly drives it (with you in frame).
- [USER] steps (see `docs/blender.md`):
  1. `uv run python scripts/build_blender_preview.py`; `uv run python scripts/package_blender_addon.py`;
     install `blender\ekagrata_live.zip` via Preferences → Add-ons → ⌄ → Install from Disk, enable it.
  2. Open `blender\ekagrata_preview.blend`, N-panel → EKAGRATA → Start.
  3. `uv run python scripts/stream_to_blender.py --session data/sessions/2026-09-29_swings01_002 --loop`:
     confirm the figure moves; paste the panel values and the script summary.
  4. `uv run python scripts/stream_to_blender.py` (camera from `camera.yaml`, now index 1) standing in frame:
     confirm it follows you; press
     Stop and paste the console summary (Window → Toggle System Console).
- Deviations from the approved plan:
  - The message has one extra field, `source: "real" | "replay"`, needed so Blender can label replay capture
    age as not meaningful (amendment 1). All other fields are exactly as specified.
  - The landmark subset is 13 names (nose + both shoulders, elbows, wrists, index, pinky, hips); the plan said
    11 by miscount.
  - Stop also prints whole-run median/p95 (in addition to the last 2 s), because the 2 s window is empty when
    the sender stopped more than 2 s before Stop.
- Known limitations:
  - **Axis mapping unverified until check_axes passes** (shown in the scene label, panel, console, docs).
  - Raw camera shadow only: segment lengths vary frame to frame; the figure's origin is MediaPipe's hip
    midpoint, lifted by a display offset (Z = 1.0 m), so it floats (no legs/feet in the subset).
  - The live run measured 8.19 fps capture (vs 29.9 fps on 2026-09-29 with the same config); cause not
    investigated in this milestone (lighting and Blender GUI load at the same time are candidates) — unverified.
  - Hips are invisible in the existing recordings (framing), so shoulder–hip links stay hidden there.
  - Ages across processes are valid only on the same machine (system-wide QueryPerformanceCounter).
  - The add-on's modal operator is exercised only by the agent GUI runs above, not by the automated suite
    (a modal TIMER needs a window; the headless tests call the same apply/drain/keep-newest functions).
- Suggested commit message (after the [USER] steps pass):
  ```
  M5-preview: Blender 5.0.1 raw camera shadow (UDP sender, replay/live streamer, scene builder, live add-on)

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

### Camera and latency investigation (M5-preview follow-up) — 2026-09-30 (measurement only; fix awaiting approval)
- SPEC: message field `source` documented in M5-preview and M5 (M5 also gets `t_send_ns`, and `"sim"` for the
  simulated candidate rig).
- (a) Measurements:
  - `check_env.py --camera 0 --seconds 10` → `Measured : 8.53 fps over 9.96 s (86 frames)`, reported fourcc
    `'}ë6ä'` (invalid). Thumbnail of index 0: black Iriun "waiting" screen, mean brightness 0.6/255.
  - Windows devices: `USB2.0 HD UVC WebCam` and `Iriun Webcam` present; `IriunWebcam` process running.
  - DirectShow index 1 = laptop webcam (sees the user; fourcc `'MJPG'`, brightness 169.9, 15.10 fps without
    exposure set). `check_env.py --camera 1 --seconds 10` → `Measured : 15.03 fps over 9.98 s (151 frames)`
    (check_env never applies exposure, so auto-exposure limits it).
  - `camera_probe.py --camera 1`: dshow 1280x720 MJPG exposure -6 → 29.90 fps, median interval 32.1 ms,
    brightness 46.8, 0 dups; auto → 15.79 fps, brightness 126.5; -7 → 29.90 fps, brightness 50.0 (probe
    suggestion). MSMF rows at index 1 show brightness 0.5 and 79–94 dups = the Iriun screen: **DirectShow and
    MSMF number the devices differently.**
- (b) Code path: `stream_to_blender.py --camera N` uses exactly the `camera.yaml` settings via the same
  `LiveCameraSource` → `CameraCapture.start()` as `record_session.py` (backend, 1280x720, fps request, MJPG set
  twice, `exposure_auto`, `exposure`), and the same `create_landmarker` / `detect` / `video_timestamp_ms`; only
  `device` is overridden by `--camera`. Differences: none in settings; streamer creates the landmarker before
  opening the camera and writes nothing to disk. `check_env.py` differs: own open code, no exposure.
  Streamer now prints the requested and driver-reported camera settings; 30 s run reported
  `{'w': 1280, 'h': 720, 'fps': 60.0, 'fourcc': 'MJPG', 'auto_exposure': -1.0, 'exposure': -6.0}`.
- (c) Latency breakdown, `stream_to_blender.py --camera 1 --seconds 30` with Blender 5.0.1 GUI receiving; the
  user was seated at the laptop in frame (not standing): `Frames with pose : 897 (100.0 %)`,
  `Capture fps : 29.89`, `Frames dropped : 0`, `Send rate : 29.95 Hz`. Stages (n = 897, ms, median / p95 / max):
  decode (retrieve) 31.93 / 47.62 / 77.82; queue wait 0.22 / 0.37 / 0.81; dequeue → inference 0.01 / 0.01 /
  0.05; inference 17.52 / 20.20 / 36.49; map + smooth 0.19 / 0.30 / 0.87; encode + sendto 0.16 / 0.23 / 0.38;
  grab → sent 50.16 / 65.49 / 115.06; grab interval 32.09. Blender (716 applied, lost 0): age since send
  8.5 / 17.8 ms; **age since capture 58.4 / 78.4 ms** (whole run). Consistency: 50.16 + 8.5 = 58.7 ≈ 58.4.
  - **Queue wait ≈ one frame interval: refuted.** Queue wait / grab interval = 0.01 (median and p95); inference
    (17.5 ms) is shorter than the frame interval (32.1 ms), so frames are consumed as soon as they are queued.
  - The ~one-interval wait is in grab → retrieve instead. Direct probe (300 frames, same settings):
    `grab()` median 0.01 ms, `retrieve()` median 32.00 ms (p95 47.80); JPEG decode of a 1280x720 frame costs
    2.85 ms median (`cv2.imdecode` reference). So with OpenCV DirectShow, `grab()` returns immediately and
    `retrieve()` waits for the next frame: **`t_host_ns` (stamped after grab) precedes the frame's arrival at
    the host by about one frame interval.** This affects all M1/M2 live recordings (camera timeline offset) and
    overstates capture age by that amount.
  - Blender send → apply 8.5 ms median is consistent with the ~60 Hz timer (up to 16.7 ms wait).
  - Scratch files (not in the repo): per-frame stamps `timing_live_30s.csv`, probes `which_cam.py`,
    `grab_probe.py` in the session scratchpad.
- Code changed (instrumentation only): `ekagrata/core/types.py` (`CameraFrame.t_retrieved_ns`, optional),
  `ekagrata/io/camera.py` (stamp after `retrieve()`), `ekagrata/transport/udp_twin.py` (`last_sent_ns` after
  `sendto`), `scripts/stream_to_blender.py` (stage stamps, `--timing-csv`, camera info line, breakdown report),
  `tests/test_camera.py`, `tests/test_udp_twin.py`, `tests/test_stream_replay.py`, `docs/blender.md` (method),
  `docs/SPEC.md` (`source` field).

### Camera guard + timestamp fix (approved 2026-09-30; M5-preview follow-up) — 2026-09-30 (DONE)
- Approved by the user: wrong-camera guard, `device: 1` default, startup camera line in every camera script,
  timestamp point "after_retrieve", `check_env.py` config mode + `--raw`, keep the E07 timing CSV. Not done
  (as instructed): no inference or Blender-timer optimisation; name-based device selection deferred to the M14
  demo-hardening checklist (open item added in SPEC M14).
- Files changed:
  - `ekagrata/io/camera.py`: `t_host_ns` stamped immediately after `retrieve()`; `t_grab_ns` kept as a
    diagnostic; `TIMESTAMP_POINT = "after_retrieve"` in `start()` info (→ `session.json camera.timestamp_point`);
    wrong-camera guard in `start()` (fourcc mismatch → RuntimeError naming both values; first
    `STARTUP_CHECK_FRAMES = 5` frames all with mean pixel value ≤ `BLACK_MEAN_MAX = 5.0` → RuntimeError with a
    "try another index" hint; no frames at startup → RuntimeError); `describe(cfg)` startup line.
  - `ekagrata/core/types.py`: `CameraFrame.t_grab_ns` (optional, replaces this milestone's `t_retrieved_ns`);
    `t_host_ns` comment updated.
  - `configs/camera.yaml`: `device: 1` (laptop webcam) with a comment on shifting indices.
  - `scripts/record_session.py`, `scripts/stream_to_blender.py`: `--camera` optional (default = `camera.yaml`
    `device`), startup camera line; record_session writes `timestamp_point` ("after_retrieve" live, "pts" file).
    Streamer stage names updated ("grab -> retrieve end", "t_host -> sent").
  - `scripts/check_axes.py`, `scripts/camera_probe.py`: startup camera line (index + backend).
  - `scripts/check_env.py`: default = `camera.yaml` settings through `CameraCapture` (incl. guard, exposure),
    reports brightness and drops; `--raw` = old behaviour (plus a fourcc-mismatch warning).
  - Tests: `tests/test_camera.py` (fake camera with startup frames; new: `test_timestamp_is_taken_after_retrieve`
    — grab() instant, retrieve() blocks 30 ms, stamp asserted after retrieve; `test_info_reports_timestamp_point`;
    `test_fourcc_mismatch_fails_naming_both`; `test_near_black_startup_frames_fail_with_hint`;
    `test_no_frames_at_startup_fails`), `tests/test_stream_replay.py` (stage names).
  - Docs: `docs/SPEC.md` (§4.1 camera timestamp point + bias of old sessions, §4.3 `timestamp_point`, M1
    `io/camera.py` text, M14 open item), `docs/conventions.md` (Time), `docs/blender.md` (method, camera index).
  - Evidence: `experiments/E07_latency/README.md`, `raw/2026-09-30_breakdown_seated_pre_fix.csv`,
    `raw/2026-09-30_breakdown_seated_post_fix.csv`.
- Old sessions (`data/sessions/*`) untouched. Their bias, as documented in SPEC §4.1: "≈ one frame interval
  early, approximately constant; velocities/accelerations unaffected; absolute cross-sensor sync affected".
- Tests: `213 passed` · Ruff: `All checks passed!`
- Guard on the real hardware: `check_env.py --seconds 5` (config, index 1) → `Camera: device index 1, backend
  dshow`, reported fourcc `'MJPG'`, `Measured : 30.04 fps over 4.99 s (151 frames)`, `Brightness : mean pixel
  value 51.2 / 255`, `Dropped : 0`. `check_env.py --camera 0` → `ERROR: camera index 0 (dshow): requested
  fourcc 'MJPG' but the driver reports '}ë6ä'. It is probably not the intended camera (e.g. the Iriun virtual
  camera can take index 0). Try another index with --camera N, or set `device` in configs/camera.yaml.`
  (The near-black branch is covered by the unit test only; on this hardware the fourcc check triggers first.)
- Post-fix 30 s breakdown (`stream_to_blender.py --camera 1 --seconds 30`, user seated in frame, Blender 5.0.1
  GUI receiving): 900 frames, pose 100 %, capture 30.00 fps, 0 dropped, send 30.01 Hz. Stages (ms, median / p95
  / max, n = 900): grab → retrieve end 31.95 / 47.61 / 49.67; queue wait 0.21 / 0.29 / 0.51; dequeue →
  inference 0.01 / 0.01 / 0.03; inference 17.51 / 19.15 / 31.46; map + smooth 0.19 / 0.28 / 0.53; encode +
  sendto 0.16 / 0.22 / 0.43; t_host → sent 18.08 / 19.76 / 32.30; frame interval 32.05; queue wait / frame
  interval 0.01. Blender (739 applied, 0 lost): age since send 8.2 / 18.0 ms; **age since capture 26.5 /
  36.2 ms** (pre-fix run: 58.4 / 78.4 ms). Consistency: 18.08 + 8.2 = 26.3 ≈ 26.5.
- Known limitations: "age since capture" now starts at frame arrival + decode; camera-internal latency
  (exposure, readout, USB, driver) and display latency are not included and are still unmeasured (M7 LED
  sync, E07). Both runs were seated, not standing. Camera indices can still shift (M14 open item).
- Suggested commit message:
  ```
  Camera guard + after_retrieve timestamps: wrong-camera guard, device 1, check_env config mode, E07 evidence

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

## Open questions for the user
- RESOLVED 2026-09-30: `--camera` is optional in `record_session.py`, `stream_to_blender.py`, `check_env.py`
  and `check_axes.py` (default = `camera.yaml` `device`).
- Git is not initialised in this folder. Should the agent run `git init` (no commit), or will you?
- 2026-09-29: phone camera tested — Realme 7 via Iriun Webcam over USB is camera index 1. Per the user, the project
  uses the laptop webcam only for now, so `configs/camera.yaml` stays at `device: 0`. User run `check_env.py --camera 1`: `Measured : 30.01 fps over 5.00 s (151 frames)`,
  `Interval ms : mean 33.32, min 26.98, max 40.85, median 33.52`; driver-reported fourcc is not a valid code
  (`'}ë6ä'`, Iriun virtual driver). Resolution/fps are set in the Iriun apps; requested values may be ignored.
- RESOLVED 2026-09-29: laptop webcam ~15 fps was caused by auto-exposure; manual exposure -6 gives ~30 fps
  measured (see "Camera diagnosis" above).
- RESOLVED 2026-09-30: the 8.19 fps live capture was DirectShow index 0 = Iriun virtual camera (see "Camera
  and latency investigation"). Fixed: see "Camera guard + timestamp fix". Note the index SHIFTED: on
  2026-09-29 Iriun was index 1, on 2026-09-30 index 0.
- RESOLVED 2026-09-30: keep the message field `source`; documented in SPEC M5-preview and M5.
