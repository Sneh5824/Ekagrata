# EKAGRATA — Software Specification (source of truth)

Version 1.0 · Scope: **software-first build** (everything that can be built and tested before the BNO055/ESP32
hardware is characterised). Read `AGENTS.md` first; its rules apply everywhere.

---

## 0. How the agent must use this document

1. Read `AGENTS.md`, then this file's §1–§6 once, then **only the section of the current milestone**.
2. Check `docs/PROGRESS.md` to find the first milestone not marked `DONE`. That is the current milestone.
3. For that milestone: write an implementation plan → wait for approval → implement → run `/verify`
   (ruff + pytest) → check every acceptance criterion with real command output → update `docs/PROGRESS.md` → STOP.
4. If the spec is ambiguous, contradictory, or technically wrong: STOP and ask. Do not guess silently.
   Record the question under "Open questions" in `docs/PROGRESS.md`.
5. Anything tagged **[USER]** needs the human (recording a video, installing Blender, standing in front of the
   camera). Prepare everything, then ask the user to do that step and paste the output back.

---

## 1. Project summary

### 1.1 What EKAGRATA is
An athlete **Digital Twin** and real-time **Motion Shadow** for one well-defined movement: the **badminton
forehand smash** (right-handed by default; handedness configurable). The system captures the athlete's arm
motion (camera + 3 wearable IMUs), reconstructs it on an athlete-specific kinematic model, mirrors it live on a
virtual athlete, simulates the resulting racket–shuttle outcome, evaluates candidate movement variations, and
reports measured, quantitative feedback.

### 1.2 Current architecture (overrides the original PDF)
- **Compute:** one Windows 11 laptop (Intel i5, 16 GB, RTX 4050). Raspberry Pi is future work only.
- **IMU nodes:** 3 × (BNO055 + ESP32 + LiPo). Node 1 `upper_arm` (lateral upper arm), node 2 `forearm`
  (distal forearm, 5–8 cm above wrist), node 3 `racket` (BNO055 on racket handle butt, ESP32 on wrist strap) —
  or `hand` (back of hand) if configured. Optional node 4 `torso` (sternum).
- **Radio:** nodes → **ESP-NOW** → ESP32 #4 **receiver** on laptop USB serial (921600 baud). The receiver
  timestamps every packet with its own clock and drives a **sync LED** visible to the camera.
- **Camera:** laptop webcam or phone as USB webcam for live use; phone 240 fps slow-motion files for offline
  analysis and validation.
- **Software:** Python 3.12 + uv on Windows. Blender (own process) for the high-quality twin. ROS 2 (M10) as
  middleware with thin adapters. MuJoCo for dynamics/RL. **Gazebo is cut** (no engineering purpose here).

### 1.3 Known errors in the original PDF (do not propagate)
- Table 6 lists "Control Signal: PWM 1100/1500/1900 µs" for the BNO055 — wrong (servo/ESC spec).
- Figure 5 CAD lists an MPU-9250 in a 45×35×16 mm case — not our hardware.
- Raspberry Pi 5 as the primary compute unit — replaced by the laptop.
- 100 Hz "chosen via Nyquist with f_max = 20 Hz" — unsupported. 100 Hz is the BNO055 fusion output ceiling; the
  achieved rate must be **measured**.
- The 4.7 cm Motion Shadow error is an **arithmetic example**, not a result. Never reuse it.
- The elbow-angle example is 2D and view-dependent; we use 3D. (Its value is 135.0°.)

### 1.4 Physical facts the design must respect
- BNO055 fusion mode is fixed at ±4 g accelerometer and ±2000 °/s gyroscope. Racket angular velocity in skilled
  smashes can far exceed 2000 °/s, so **saturation is expected on the distal node** and must be detected, flagged,
  and handled (inflate IMU noise, rely on camera/post-swing correction). Never silently assume it doesn't happen.
- The acceleration phase of a skilled smash is only tens of milliseconds, so a 30 fps camera sees ~1 blurred
  frame of it. The camera is for posture, torso, position and drift correction; IMUs are for the fast phase.
- Forearm pronation/supination is **not observable** from pose landmarks. Camera-only pronation = NaN.

---

## 2. System architecture

```
Sources (swappable)            Core pipeline (ekagrata/, no ROS)                    Consumers
---------------------          ------------------------------------------         -------------------------
LiveCamera / VideoFile  ─┐     vision: PoseLandmarker → landmark_map (→ world)      Blender twin (UDP)
                         ├──►  sync: clocks → t_sync_ns                        ──► RViz (M10, /joint_states)
Serial / Sim / Replay   ─┘     kinematics: model, FK, IK, s2s calibration          Plots / reports
IMU                            fusion: IMU-only, complementary, EKF                Session files (CSV/Parquet)
                               analysis: segmentation, features, stats            Optimizer / ML / RL
                               sim: IMU sim, camera sim, racket+shuttle, objective
```

**Hard rule:** real, simulated and replayed data all produce the same types (`ImuSample`, `PoseFrame`) and the
same files. Switching source is a config/CLI flag, never a code change.

### 2.1 Package map
| Path | Responsibility | Introduced |
|---|---|---|
| `ekagrata/core/` | `quat.py`, `timebase.py`, `types.py`, `config.py` (YAML loading + validation) | M0 (config M1) |
| `ekagrata/io/` | `camera.py`, `session.py` (session folder read/write), `protocol.py`, `serial_rx.py` | M1, M6 |
| `ekagrata/sources/` | `imu.py` (ImuSource + Serial/Sim/Replay), `pose.py` (Live/VideoFile) | M1, M6 |
| `ekagrata/vision/` | `pose_landmarker.py`, `landmark_map.py`, `led_detect.py` | M1, M2, M7 |
| `ekagrata/kinematics/` | `angles.py`, `model.py`, `fk.py`, `ik.py`, `generators.py`, `s2s_calibration.py` | M2, M4, M7 |
| `ekagrata/analysis/` | `filters.py`, `derivatives.py`, `segmentation.py`, `features.py`, `stats.py` | M2, M3, M7 |
| `ekagrata/sim/` | `smash_synth.py`, `imu_sim.py`, `camera_sim.py`, `racket.py`, `shuttle.py`, `objective.py` | M6–M8 |
| `ekagrata/fusion/` | `sync.py`, `imu_only.py`, `complementary.py`, `ekf.py` | M7 |
| `ekagrata/optim/` | `params.py`, `search.py` (random, DE, optuna), `report.py` | M8 |
| `ekagrata/transport/` | `udp_twin.py` | M5 |
| `ekagrata/pipeline/` | `live.py` (stage graph + per-stage latency stamps) | M9 |
| `ekagrata/ml/` | `dataset.py`, `models.py`, `evaluate.py` | M12 |
| `ekagrata/rl/` | `env.py`, `train.py` | M13 |
| `ros2_ws/src/` | `ekagrata_msgs`, `ekagrata_description`, `ekagrata_nodes`, `ekagrata_bringup` | M10 |
| `firmware/` | `common/packet.h`, `imu_node/`, `receiver/` | M11 |
| `blender/` | `scripts/build_scene.py`, `scripts/play_motion.py`, `addon/ekagrata_live/` | M5 |
| `experiments/` | `EXX_name/{protocol.md, run.py, results/}` | M7 onward |

---

## 3. Conventions (in addition to AGENTS.md §4)

### 3.1 Segments, frames and joints
Segment frames follow ISB style: **+Y along the segment's long axis pointing proximally** (toward the parent
joint), +X anterior at the reference pose, +Z = X × Y. Reference ("zero") pose: standing, arm hanging, palm facing
the thigh, racket pointing down along the forearm axis.

| Joint | Parent → child | Stored primary representation | Derived angles (rad) |
|---|---|---|---|
| shoulder | torso → upper_arm | `q_torso_upper_arm` | `shoulder_elev` ∈ [0, π] (0 = arm down), `shoulder_plane` (0 = pure abduction, +π/2 = forward flexion), `shoulder_rot` (humeral axial rotation, ISB Y-X-Y third angle; NaN within 10° of elev 0 or π) |
| elbow | upper_arm → forearm | `q_upper_arm_forearm` | `elbow_flex` ∈ [0, π] (0 = straight), `forearm_pron` = **twist** of the swing–twist decomposition about the forearm Y axis |
| wrist | forearm → racket (or hand) | `q_forearm_racket` | `wrist_flex`, `wrist_dev` from the **swing** part, projected onto forearm X/Z; `racket_twist` = twist part |

Swing–twist decomposition is implemented once in `core/quat.py` (`swing_twist(q, axis)`), with tests.
Left-handed players: mirror across the torso sagittal plane at ingestion; all downstream code assumes right-handed.

### 3.2 Camera-only observability
From pose landmarks: torso frame, `shoulder_elev`, `shoulder_plane`, `elbow_flex` are observable;
`shoulder_rot` only when `elbow_flex > 15°`; `forearm_pron` is NaN; wrist angles are low-confidence (pose model
hand points only) and flagged `quality="low"`.

### 3.3 Coordinates and units recap
World: right-handed, Z up, metres. Court frame (for simulation): origin at the centre of the player's baseline,
+X toward the net, +Z up. Quaternions [w,x,y,z]. Time int64 ns. Angles rad (display `_deg`).

---

## 4. Data formats

### 4.1 Session folder
```
data/sessions/<YYYY-MM-DD>_<session>_<athlete>/
  session.json                    # metadata (schema below); written at start, finalised at end
  video/cam0.mp4                  # optional raw video
  video/cam0_frames.csv           # frame_idx, t_host_ns, t_sync_ns, pts_ns (file mode), dropped_before
  pose/cam0_landmarks.parquet     # frame_idx, t_sync_ns, lm{i}_{x,y,z,vis,pres} (i=0..32, image-normalised),
                                  #   wlm{i}_{x,y,z} (MediaPipe world, metres, raw MediaPipe axes)
  imu/imu.csv                     # one row per sample, all nodes (columns in §4.2)
  imu/serial_raw.bin              # optional raw serial bytes (replayable)
  sync/events.csv                 # LED/sync events: t_rx_us, t_host_ns, blink_id, source
  motion/joints.parquet           # t_sync_ns, source(cam|imu|fused|sim_truth), joint quats + derived angles,
                                  #   their _vel and _acc, racket_head_{x,y,z,vx,vy,vz}, quality flags
  strokes/strokes.csv             # stroke_id, t_start_ns, t_bs_ns, t_fs_ns, t_imp_est_ns, t_end_ns, label, notes
  strokes/features.parquet        # per-stroke features (M3)
  trials.csv                      # stroke_id, label, shuttle_speed_mps, speed_method, landing_x_m, landing_y_m, rater
  truth/                          # simulation ground truth only (never present for real sessions)
```
Raw files are append-only. Derived files may be regenerated; each derived file records the code version
(`git describe --always --dirty`) in `session.json.derived[]`.

### 4.2 `imu/imu.csv` columns
`t_sync_ns, t_node_us, t_rx_us, t_host_ns, node, seq, qw, qx, qy, qz, gx, gy, gz, ax, ay, az, lax, lay, laz,
cal_sys, cal_g, cal_a, cal_m, flags, source` — gyro rad/s, acc m/s² (specific force, sensor frame),
`lax..laz` linear acceleration (gravity removed), `flags` bitfield per §4.4, `source` ∈ {real, sim, replay}.

### 4.3 `session.json` (minimum keys)
`schema_version, session_id, athlete_id, created_utc, handedness, code_version, host {os, python, packages},
camera {device, backend, requested {w,h,fps}, reported {w,h,fps}, measured_fps}, model {name, file_sha256},
imu {nodes: [{node, placement, mounting_note, firmware, bno_mode}], receiver_port}, calibration {…},
clock_sync {per-node offset/drift, method}, notes, derived []`.

### 4.4 Binary protocol (firmware ↔ Python; single source: `firmware/common/packet.h` mirrored in
`ekagrata/io/protocol.py`, little-endian, packed)
**NodePacket (40 bytes):** `u8 version(=1) | u8 node_id (1 upper_arm, 2 forearm, 3 racket/hand, 4 torso) |
u16 flags | u32 seq | u32 t_node_us | i16 quat[4] (w,x,y,z; /16384) | i16 gyro[3] (/16 → °/s → rad/s) |
i16 acc[3] (/100 m/s²) | i16 lin_acc[3] (/100 m/s²) | u8 calib (sys<<6|gyro<<4|acc<<2|mag) | u8 reserved`.
Flags: bit0 duplicate, bit1 gyro_saturated, bit2 acc_saturated, bit3 calib_restored, bit4 simulated.
(Scales are BNO055 register scalings; confirm against the datasheet in M11.)

**Serial frame (receiver → laptop):** `0xAA 0x55 | u8 body_len | body | u16 crc16` (CRC-16/CCITT-FALSE over body).
Body types (first byte): `0x01 IMU`: `u8 type | u32 t_rx_us | i8 rssi | u8 reserved | NodePacket(40)` (47 bytes).
`0x02 SYNC`: `u8 type | u32 t_rx_us | u32 blink_id` (9 bytes). `0x03 STATS`: `u8 type | u32 t_rx_us |
u32 rx_count[4] | u32 crc_fail | u32 overflow` (29 bytes).
Parser must resynchronise after garbage, handle partial frames, count CRC failures, and unwrap `u32` µs clocks.

---

## 5. Configuration files (`configs/`, YAML, validated on load with clear errors)
- `athlete/<id>.yaml`: id, handedness, mass_kg, height_m, segment lengths (upper_arm, forearm, hand) measured
  with tape [USER], shoulder width, notes.
- `racket.yaml`: length_m (≤ 0.68 per BWF), grip_to_head_centre_m, mass_kg, head width/height.
- `kinematics/arm_model.yaml`: segments, parent, joint type/axes, limits (rad), reference pose; lengths pulled from
  the athlete file.
- `camera.yaml`: device index, backend, requested resolution/fps, MediaPipe model variant, `mp_to_world`
  3×3 matrix (see M2), LED ROI.
- `sensors.yaml`: node ids ↔ segment, expected MAC addresses, BNO055 mode (IMUPLUS default), mounting notes.
- `sim/*.yaml`: IMU noise/saturation/timing, camera-sim noise, shuttle/impact parameters, objective weights.

---

## 6. Global quality bar
- Every numeric function has known-answer tests. Every file format has a write→read round-trip test.
- Every script has `--help`, validates inputs, and fails with a clear message (never a bare traceback for user
  errors such as a missing camera or model file).
- No hidden global state; config is passed explicitly. Deterministic given a seed.
- Performance numbers (fps, latency, error) are only ever printed from measurements.

---

## 7. Milestones

Order: M0 → M14. Each milestone is independently testable. **Gate** = what must be true before the next one.
M11 (firmware) may be pulled forward when the user says hardware has arrived.

---

### M0 — Foundation
**Goal:** uv project, conventions in code, environment check.
**Deps:** numpy, scipy, pandas, pyarrow, matplotlib, mediapipe, pyserial, pyyaml; dev: pytest, ruff.
Exactly one OpenCV distribution installed (check what mediapipe pulls; do not add a second one).
**Build:**
- `pyproject.toml` (Python ==3.12.*, ruff line length 110, pytest testpaths=tests), `.gitignore`
  (`.venv/ data/ models/ __pycache__/ *.parquet *.mp4 .ruff_cache/ .pytest_cache/`), `README.md`.
- `core/quat.py`: `normalize, conj, multiply, rotate_vector, from_scipy, to_scipy, from_rotvec, to_rotvec,
  relative(q_world_parent, q_world_child) -> q_parent_child, angle_between (double-cover safe), slerp,
  swing_twist(q, axis) -> (q_swing, q_twist), twist_angle(q, axis)`. Vectorised for (4,) and (N,4).
- `core/timebase.py`: `now_ns, unwrap_u32, ns_to_s`.
- `core/types.py`: frozen dataclasses `ImuSample`, `PoseFrame` (fields per §4.2 / §4.1).
- `scripts/check_env.py`: versions, installed OpenCV distributions, open camera (DSHOW then MSMF), request
  1280×720@60 MJPG, print backend, reported w/h/fps, and **measured** fps over N s from `perf_counter_ns`
  (mean/min/max interval ms, count of intervals > 1.5× median). Args `--camera --seconds --width --height --fps`.
- `docs/conventions.md` (human copy of conventions + worked example of `relative()`), `docs/PROGRESS.md`.
- Keep `docs/reference/Ekagrata_original_proposal.pdf` (already provided) — reference only.
**Tests:** SciPy round-trips (ordering asserted explicitly), 90° about Z maps (1,0,0)→(0,1,0), composition law,
`relative` identity and 30° cases, `angle_between(q,-q)=0`, slerp endpoints/midpoint, seeded random comparisons
vs SciPy, swing–twist recomposition (`swing ⊗ twist == q`) and pure-twist/pure-swing cases, `unwrap_u32` across
one and two wraps.
**Acceptance:** fresh `uv sync` works; ruff + pytest pass; `check_env.py` prints measured fps [USER runs].

---

### M1 — Camera capture + MediaPipe recorder
**Goal:** record sessions with trustworthy timestamps; process live camera or video files identically.
**Build:**
- `core/config.py`: load + validate YAML (dataclasses; clear error messages). Create `configs/camera.yaml`.
- `scripts/download_models.py`: downloads to `models/` and records SHA-256:
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{lite|full|heavy}/float16/latest/pose_landmarker_{lite|full|heavy}.task`
- `io/camera.py`: capture thread that stamps each frame with `now_ns()` **immediately after grab**, puts
  `(frame, t_host_ns, frame_idx)` into a bounded queue (drop-oldest, count drops). Backend selectable.
- `sources/pose.py`: `LiveCameraSource` and `VideoFileSource` (timestamps from container PTS; for 240 fps phone
  files, verify PTS spacing and report it). Same output type.
- `vision/pose_landmarker.py`: Tasks API `PoseLandmarker`, **VIDEO** running mode (strictly increasing ms
  timestamps derived from `t_host_ns` or PTS), variant selectable (lite/full/heavy), returns `PoseFrame` with
  image landmarks (x, y, z, visibility, presence) and world landmarks.
- `io/session.py`: create session folder, write `session.json`, `video/cam0.mp4` (optional), `cam0_frames.csv`,
  `pose/cam0_landmarks.parquet` (buffered writes; finalise on Ctrl+C without data loss).
- `scripts/record_session.py`: `--athlete --session --camera | --video PATH --model full --save-video
  --preview --seconds`. Preview window with skeleton overlay and live stats (capture fps, inference ms, queue
  drops). On exit prints a **measured** summary: frames captured, processed, dropped, capture fps, inference
  time mean/p95, detection rate (% frames with a pose).
- `scripts/benchmark_pose.py`: runs lite/full/heavy on the same recorded file and prints measured inference
  ms (mean, p95) and detection rate per variant.
**Tests:** session write→read round-trip; timestamps strictly increasing; VideoFileSource on a small synthetic
video generated in the test (cv2.VideoWriter); PoseLandmarker wrapper test skipped if model file absent.
**Acceptance:** [USER] records 30 s live and processes one phone video; summaries printed from measurements;
parquet loads with the documented columns.
**Gate:** trustworthy timestamps and landmark files exist.

---

### M2 — Landmark mapping, joint angles, filtering, derivatives
**Goal:** 3D joint angles (with correct observability) and their velocities/accelerations from camera data.
**Build:**
- `vision/landmark_map.py`: converts MediaPipe world landmarks to EKAGRATA world (Z up) with the `mp_to_world`
  matrix from `camera.yaml`; asserts det = +1 (reject mirroring unless handedness mirroring is intended).
- `scripts/check_axes.py` [USER]: live test — "raise right arm forward / to the side / up"; prints which world
  axis the right wrist moved along, so the user can confirm the mapping. Document the confirmed matrix.
- `kinematics/angles.py`:
  - torso frame from landmarks (mid-hip, mid-shoulder, shoulder line) per §3.1;
  - camera-based segment frames for upper arm (long axis + arm-plane normal; NaN if elbow_flex < 15°);
  - derived angles per §3.1/§3.2, with NaN where unobservable and `quality` flags;
  - from-quaternion path: given `q_world_seg` for torso/upper_arm/forearm/racket (IMU path, used from M7),
    compute joint quaternions via `quat.relative` and the same derived angles. **One definition, two inputs.**
- `analysis/filters.py`: visibility gating (vis < threshold → NaN), gap interpolation ≤ `max_gap_frames`,
  offline zero-phase Butterworth (`filtfilt`, cutoff configurable), online One-Euro filter.
- `analysis/derivatives.py`: resample to a uniform grid on `t_sync_ns`, then Savitzky–Golay derivative;
  NaN-aware (no derivative across gaps).
- `scripts/compute_joints.py --session …` → `motion/joints.parquet` (`source=cam`) + plots.
**Tests:** synthetic skeletons with known angles (arm down → elev 0; horizontal abduction → elev π/2, plane 0;
forward flexion → plane π/2; elbow straight → 0, 90° bend → π/2); invariance of joint angles to a global rotation
of all landmarks; derivative of a known sinusoid within tolerance; NaN propagation; filter phase lag ≈ 0 for
filtfilt.
**Acceptance:** [USER] 10 recorded smash-like swings → joint angle and angular-velocity plots; pronation column is
NaN for camera; tests pass.

---

### M3 — Stroke segmentation, features, labelling, shuttle-speed tool
**Goal:** turn continuous recordings into labelled strokes with features and outcome labels.
**Build:**
- `analysis/segmentation.py`: detect strokes from wrist (or racket, later) speed with hysteresis thresholds and
  minimum separation; phases: `t_start`, backswing start `t_bs`, forward swing start `t_fs`, estimated impact
  `t_imp_est` (peak distal speed; name it *estimated*), `t_end`. Parameters in config.
- `analysis/features.py` per stroke: peak angular velocity per joint and its time relative to `t_imp_est`,
  proximal-to-distal peak ordering, max elbow extension velocity, elbow angle at `t_imp_est`, estimated contact
  height (wrist z − shoulder z), stroke duration, forward-swing duration, peak wrist speed. Output
  `strokes/features.parquet`.
- `scripts/label_strokes.py`: plays each detected stroke clip (from video) and asks for label
  (smash/clear/drive/other/reject) → `strokes/strokes.csv`.
- `scripts/measure_shuttle.py` [USER]: for 240 fps video — user clicks two scale points of known distance
  (e.g. court line), then clicks the shuttle in ≥ 4 consecutive post-impact frames; computes speed by linear fit
  of position vs PTS time; writes `trials.csv` with `speed_method="planar_click_240fps"` and prints the
  **stated assumptions** (flight roughly parallel to the image plane; scale plane = flight plane).
- `analysis/stats.py`: mean, SD, median, p95, max, bootstrap 95% CI, Bland–Altman (bias, limits of agreement).
**Tests:** segmentation on synthetic speed profiles (known number of strokes and phase times ± 1 sample);
feature extraction on a synthetic stroke with known peaks; shuttle-speed math on synthetic clicks; stats vs
hand-computed values.
**Acceptance:** [USER] session with ≥ 10 swings → strokes auto-detected, labelled, features written.

---

### M4 — Kinematic model, FK, IK, model generators
**Goal:** one athlete-specific kinematic model that drives everything (Python, Blender, RViz, MuJoCo).
**Build:**
- `configs/kinematics/arm_model.yaml` + `configs/athlete/example.yaml` + `configs/racket.yaml`.
- `kinematics/model.py`: load; joints = shoulder (3 DoF, represented as quaternion), elbow flex, forearm pronation,
  wrist flex, wrist deviation, racket rigid offset from hand/grip; limits in rad.
- `kinematics/fk.py`: joint state (+ torso pose) → `q_world_seg`, joint centre positions, racket head centre and
  racket face normal; vectorised over time.
- `kinematics/ik.py`: fit joint state to camera world landmarks (shoulder, elbow, wrist, hand points) with fixed
  segment lengths using `scipy.optimize.least_squares`, joint limits as bounds, warm start from previous frame,
  landmark weights from visibility; unobservable DoFs (pronation) regularised to previous value and flagged.
  Output: kinematically consistent reconstruction (`source=cam_ik`).
- `kinematics/generators.py`: from the same model emit (a) `generated/arm.urdf`, (b) `generated/arm.xml` (MJCF,
  with segment masses/inertias from de Leva (1996) proportions scaled by athlete mass — cite in docstring),
  (c) `generated/rig.json` for Blender (bone names, heads, tails, rolls/axes, parents).
**Tests:** FK zero pose = arm hanging with hand at shoulder − (upper_arm + forearm + hand) along −Z;
known-pose endpoints; FK→IK round trip recovers observable DoFs (tolerance stated); URDF and MJCF parse as XML
(MJCF load test with mujoco is skipped until M8b); segment lengths identical across all three outputs.
**Acceptance:** IK reconstruction of an M2 session saved; FK endpoints of the same state agree across Python
and generated models.

---

### M5 — Blender Digital Twin + UDP live link
**Goal:** the visual twin: athlete arm + racket + court + shuttle, driven from files or live.
**[USER]:** install Blender (4.5 LTS or newer) and add it to PATH; the agent checks the version and uses only
APIs available in that version.
**Build:**
- `blender/scripts/build_scene.py` (run: `blender --background --python … -- --rig generated/rig.json --out
  blender/ekagrata_twin.blend`): armature from `rig.json` (bone rest pose = model zero pose; bone local axes
  mapped to segment frames per §3.1 and documented), simple capsule meshes, racket mesh, singles court
  (13.40 × 5.18 m lines), net (1.524 m at centre), shuttle object, camera + lights, and a second semi-transparent
  **"candidate" armature** for side-by-side comparison.
- `scripts/export_for_blender.py`: `motion/joints.parquet` → plain CSV of per-bone local quaternions per frame
  (Blender's Python may not have pandas/pyarrow).
- `blender/scripts/play_motion.py`: keyframes the armature from that CSV (athlete and/or candidate).
- `transport/udp_twin.py`: sender — compact message `{seq, t_sync_ns, rig: "athlete"|"candidate",
  bones: {name: [w,x,y,z]}, racket_head: [x,y,z]}` as JSON (fallback: struct) to `127.0.0.1:9870`.
- `blender/addon/ekagrata_live/`: modal timer operator (≈ 60 Hz) with a non-blocking UDP socket; drains the
  socket and applies **only the newest** message; panel shows receive rate, message age (ms), drops; clean
  start/stop; never blocks the UI thread.
- `blender/tests/smoke.py` (headless): builds the scene, applies a known pose, asserts the hand/racket endpoint
  equals Python FK within 1 mm.
**Tests:** message encode/decode round-trip; smoke test run if Blender is on PATH (skip otherwise, clearly).
**Acceptance:** [USER] recorded stroke plays back in Blender; live mode follows the camera IK stream; the add-on
panel shows measured receive rate and message age.
**Gate:** Digital Twin driven by real camera data.

---

### M6 — Binary protocol, IMU sources, IMU simulator
**Goal:** develop the whole IMU path before hardware; swapping fake → real is a CLI flag.
**Build:**
- `firmware/common/packet.h` and `io/protocol.py` implementing §4.4 exactly (a test asserts `struct.calcsize`
  == 40 / 47 / 9 / 29 and that `packet.h` field order matches — parse the header or keep a shared table).
- `io/serial_rx.py`: streaming frame parser (resync on garbage, partial frames, CRC counting), converts to
  `ImuSample` (SI units), unwraps `t_node_us` / `t_rx_us`, attaches `t_host_ns` at read time; optional raw-bytes
  tee to `imu/serial_raw.bin`.
- `sources/imu.py`: `ImuSource` protocol with `read_available() -> list[ImuSample]`, implementations
  `SerialImuSource(port)`, `SimulatedImuSource(config)`, `ReplayImuSource(csv | raw.bin, speed=1.0|max)`.
- `sim/smash_synth.py`: parametric smash joint trajectories (minimum-jerk segments; phase timings and peak
  velocities from config) **labelled illustrative, not measured**; plus "from recording": take an M4 IK
  reconstruction as the truth trajectory.
- `sim/imu_sim.py`: from truth `q_world_seg(t)`, `p_world_sensor(t)` and mounting `q_seg_sensor` + lever arm:
  - gyro = body-frame angular velocity (`2·conj(q)⊗q̇`, vector part); specific force = `R(q)ᵀ(a_world − g_world)`
    with `g_world = (0,0,−9.81)`; linear acceleration = `R(q)ᵀ a_world`;
  - noise: white noise density, bias instability/random walk, initial bias, scale error (config);
  - **saturation** clip at ±2000 °/s and ±4 g (fusion-mode ranges) with flags set;
  - BNO055-like orientation output: run a Mahony/Madgwick filter on the noisy clipped data at 100 Hz, so
    saturation-induced orientation error emerges naturally (document that this is an approximation of Bosch's
    closed-source fusion);
  - timing: node clock offset + drift (ppm), 200 Hz polling of 100 Hz fusion output (duplicates flagged),
    Bernoulli + burst packet loss, radio latency distribution, receiver clock;
  - output as `ImuSample`s **and** as encoded serial bytes (so the real parser is exercised).
- `sim/camera_sim.py`: truth FK landmarks → 30 fps (configurable) with exposure-window averaging (blur),
  Gaussian noise, and speed-dependent dropout.
- `scripts/sim_session.py`: writes a complete simulated session folder (`source=sim`, `truth/` included).
**Tests:** protocol round-trip for every body type, CRC failure detection, resync after random garbage, split
frames across reads; sim gyro of a constant-rate rotation equals that rate; static sensor reads +9.81 on the
up-axis; saturation flags set exactly where |ω| > limit; duplicate/loss rates match config within statistical
tolerance (seeded); `ReplayImuSource(raw.bin)` reproduces the same samples as the simulator emitted.
**Acceptance:** a simulated session is produced and replays through `SerialImuSource`'s parser path unchanged.

---

### M7 — Time sync, sensor-to-segment calibration, fusion (validated against simulated truth)
**Goal:** the fusion pipeline, with honest evaluation — all results in this milestone are **SIMULATED**.
**Build:**
- `fusion/sync.py`: per-node map `t_node_us → t_rx_us` (robust linear fit on the lower envelope of
  `t_rx − t_node`; estimates offset + drift), `t_rx_us → t_host_ns` map from serial arrival times, camera offset
  from LED events, final `t_sync_ns` for every sample; reports residuals.
- `vision/led_detect.py`: LED onset detection in a configurable ROI (brightness step with hysteresis) → camera
  timestamps of blinks; matched to receiver `SYNC` frames → camera offset estimate + residuals.
- `kinematics/s2s_calibration.py`: (1) static reference pose (§3.1 zero pose) → `q_seg_sensor` per node, with
  heading taken from the torso (camera) or torso IMU; (2) functional elbow-hinge axis estimation from gyro data
  of flex/extend motions (Seel et al., 2014 — cite); returns calibration + quality metrics; stored in
  `session.json.calibration`.
- `fusion/imu_only.py`: calibrated `q_world_seg` per node → joint quaternions/angles via M2's quaternion path.
  Shoulder requires a torso orientation: from camera (default) or torso node (if configured).
- `fusion/complementary.py`: IMU orientation at IMU rate; slow heading (yaw) correction of each IMU from the
  camera when camera confidence is high and motion is slow; camera provides torso and root position; saturation
  flags suspend correction trust windows.
- `fusion/ekf.py` (M7b, after the above passes): state = joint DoFs + rates; process = constant-velocity;
  measurements = IMU joint rotations (100 Hz, noise inflated when flagged) and camera landmarks through FK
  (30 Hz, noise from visibility). Handle out-of-order/late measurements by timestamp.
- `experiments/E_SIM_fusion/`: camera-only vs IMU-only vs complementary vs EKF against truth, per joint and per
  phase (slow / forward swing / follow-through), with and without saturation; RMSE, MAE, p95, max using
  `analysis/stats.py`; every plot and table titled **"SIMULATED"**.
**Tests:** clock fit recovers injected offset/drift within tolerance; LED detection on synthetic frames;
calibration recovers injected mounting rotations (report angular error); IMU-only fusion with perfect sensors
equals truth to numerical precision; complementary filter removes an injected yaw drift.
**Acceptance:** E_SIM_fusion report generated, reproducible from a seed, clearly labelled simulated.
**Gate:** the only thing missing for real fusion is real hardware.

---

### M8 — Smash simulation, objective, what-if and optimization
**Goal:** evaluate candidate movement variations under an explicit, versioned model.
**Build:**
- `sim/racket.py`: racket head velocity and face normal at `t_imp_est` from FK.
- Impact model: shuttle launch speed = `k_impact × racket_head_speed` (default k from `sim/impact.yaml`, with a
  literature-derived default and its source cited in the YAML comment), launch direction from the face normal
  blended with head-velocity direction (blend weight in config). Documented as a model assumption.
- `sim/shuttle.py`: `dv/dt = g − (g / v_t²)·|v|·v`, `v_t` default 6.5 m/s (cite source in YAML), integrated with
  `solve_ivp` + ground event; outputs landing point, flight time, net-plane crossing height, speed at net,
  descent angle at landing. Court per §3.3; player position from config.
- `sim/objective.py`: versioned objective `J = w_speed·v0 + w_steep·descent_angle − w_land·|landing − target| −
  penalties` (net fault, out of court, joint-limit violation, implausibility). Weights in config; the objective
  version is written to every result file.
- `optim/params.py`: candidate = the athlete's recorded stroke modified by a small parameter vector: per-joint
  peak-timing shifts (ms), per-joint amplitude scales, global time scale (≤ ~8–10 params). Bounds from the
  athlete's own observed range across trials (percentiles in config). Implausibility penalty = Mahalanobis
  distance of candidate features from the athlete's trials (when ≥ N trials; otherwise bounds only).
- `optim/search.py`: random search (**mandatory baseline**), `scipy.optimize.differential_evolution`, Optuna
  (TPE and GP samplers) — **equal evaluation budgets, multiple seeds**; every evaluation logged to CSV.
- `optim/report.py`: baseline vs best candidate (objective and components), one-at-a-time sensitivity sweeps,
  convergence curves (mean ± SD over seeds), and the sentence template: "Under objective vX and the stated model
  assumptions, candidate C scored higher than the recorded stroke." Never "optimal/perfect".
- Blender: candidate armature plays alongside the athlete (M5 tools).
- **M8b (optional, after M8 passes):** add `mujoco`; load `generated/arm.xml`; inverse dynamics (`mj_inverse`)
  → joint torques for baseline and candidates; constraint: peak torque ≤ athlete's max observed × margin.
**Deps:** optuna (M8), mujoco (M8b).
**Tests:** vacuum case (`v_t → ∞`) matches the analytic parabola; with drag, horizontal speed decays
monotonically and approaches terminal behaviour; objective monotonic in each weighted term; parameter bounds
respected; identical seeds → identical results; random search runs with the same budget as the others.
**Acceptance:** optimization report on a real recorded stroke (camera-based) + synthetic stroke, with the
model-assumption section filled in.

---

### M9 — Live pipeline (no ROS yet) with latency instrumentation
**Goal:** one command runs the whole chain live or from a recording, and measures itself.
**Build:**
- `pipeline/live.py`: stages (capture → pose → IK/angles → IMU ingest → sync → fusion → twin UDP → metrics) in
  threads/processes with bounded queues; every item carries per-stage `t_in_ns/t_out_ns`; drop policy explicit.
- `scripts/run_live.py --camera 0 --imu sim|serial:COM5|replay:<session> --twin udp --record`.
- Replay mode runs a recorded session at real-time speed through the identical code path (demo fallback).
- Live overlay/metrics window: per-stage latency (median, p95), fps, IMU unique-sample rate, packet loss,
  saturation %, camera detection rate — all measured.
- `experiments/E07_latency/protocol.md`: glass-to-glass method (240 fps phone filming the physical motion and the
  screen together) + `analyze.py` for frame-count entry.
**Tests:** pipeline with fake sources for N seconds: no unbounded queue growth, latency stamps monotonic,
clean shutdown.
**Acceptance:** [USER] 5-minute live run (camera + simulated IMU) with per-stage latency report; replay of the
same session produces identical fused output (bitwise or within float tolerance, stated).

---

### M10 — ROS 2 integration (thin adapters)
**Goal:** ROS 2 as middleware where it adds real value: record/replay (rosbag2), URDF + tf2 + RViz Motion Shadow,
process isolation, launch-time source swapping. **Core logic stays in `ekagrata/`.**
**Step 1 — spike [USER + agent], before any node code:** ROS 2 **Lyrical** (LTS; Windows 11 is Tier 1) installed
natively on Windows per the official instructions. Checklist script `scripts/ros_spike_check.ps1` + report:
talker/listener works; `rclpy` and `mediapipe` import in the same environment (Python version match); a custom
message package builds; RViz2 starts; pyserial opens a port. If mediapipe cannot coexist, the pose node runs in
the uv env and publishes via a UDP→ROS bridge node. If custom messages fail to build, fall back to
`sensor_msgs/PointCloud2` for landmarks. Fallback platform only if native fails: WSL2 + Fast DDS discovery server
+ usbipd-win. **Stop after the spike and report.**
**Step 2 — packages:**
- `ekagrata_msgs`: `Landmarks.msg` (header, float32[] x,y,z,visibility), `StrokeEvent.msg`; action
  `Optimize.action` (goal: session, stroke_id, objective_version, budget; feedback: best_so_far; result: path).
- `ekagrata_description`: URDF from M4 generator + meshes + RViz config.
- `ekagrata_nodes` (each a thin wrapper around `ekagrata/`):
  | Node | Pub/Sub | Message |
  |---|---|---|
  | `imu_bridge` (ONE node for the one receiver) | pub `/ekagrata/imu/{upper_arm,forearm,racket}`, `/diagnostics` | `sensor_msgs/Imu`, `diagnostic_msgs/DiagnosticArray` |
  | `pose_node` | pub `/ekagrata/camera/landmarks` | `ekagrata_msgs/Landmarks` |
  | `fusion_node` | sub imu + landmarks; pub `/joint_states`, `/ekagrata/racket/twist` | `sensor_msgs/JointState`, `geometry_msgs/TwistStamped` |
  | `robot_state_publisher` ×2 | `/tf` (athlete, and `candidate/` prefix) | standard |
  | `stroke_detector` | pub `/ekagrata/events/stroke` | `ekagrata_msgs/StrokeEvent` |
  | `twin_udp_bridge` | sub `/joint_states` → UDP to Blender | — |
  | `optimizer_server` | action `/ekagrata/optimize` | `ekagrata_msgs/Optimize` |
  | `sim_server` | service `/ekagrata/simulate` | candidate → outcome |
  `header.stamp` = `t_sync_ns` of the sample (never `now()` at publish). IMU topics use sensor-data QoS.
- `ekagrata_bringup`: `live.launch.py`, `replay.launch.py` (rosbag2), `sim.launch.py` (simulated IMU).
**Tests:** launch `sim.launch.py`, record a bag, replay it, compare fused `/joint_states` against the direct
Python pipeline on the same input (tolerance stated).
**Acceptance:** RViz shows the athlete + candidate shadows from a bag; determinism test passes.

---

### M11 — Firmware (compile-only until hardware is characterised)
**Target:** Arduino-ESP32 core **3.x** (note: ESP-NOW receive callback is
`void cb(const esp_now_recv_info_t *info, const uint8_t *data, int len)` in 3.x; the 2.x MAC-pointer signature
will not compile).
- `firmware/imu_node/`: `NODE_ID` via build flag; BNO055 over I²C at 100 kHz with an increased timeout (BNO055
  clock stretching; UART mode is the documented fallback); mode IMUPLUS by default (NDOF selectable); restore
  calibration offsets from NVS (`Preferences`) and save when fully calibrated; `esp_timer` at 200 Hz polling;
  skip unchanged quaternion (duplicate), set saturation flags from raw gyro/acc near range limits; fill
  `NodePacket` (§4.4); ESP-NOW unicast to the receiver MAC on a fixed channel; status LED; serial debug behind a
  flag.
- `firmware/receiver/`: ESP-NOW receive → lock-free ring buffer → serial frames (§4.4) at 921600; sync LED blink
  every N s with a `SYNC` frame; `STATS` frame every 1 s.
- `firmware/README.md`: wiring (BNO055 VIN/GND/SDA/SCL, ADR, PS pins), power notes (LiPo → 3.3 V regulation;
  TP4056 charge-current resistor must match cell capacity), flashing steps, how to read MACs.
**Acceptance (now):** both sketches compile [USER, Arduino IDE or arduino-cli]. **Acceptance (hardware):** E0
report from `scripts/imu_link_report.py`: unique sample rate, interval jitter, packet loss, CRC failures per node
over 10 min — measured.

---

### M12 — ML pipeline (code-complete; results only from real data)
**Deps:** scikit-learn.
- `ml/dataset.py`: builds a stroke table (features + label + athlete id + outcome) from sessions.
- `ml/models.py`: RandomForest, SVM (with scaling), HistGradientBoosting; regression variants for shuttle speed.
- `ml/evaluate.py`: **leave-one-subject-out** (`LeaveOneGroupOut`) — never random splits across strokes of the
  same athlete; baselines: majority class / mean predictor / racket-speed-only linear regression; metrics:
  macro-F1 + confusion matrix; MAE + R²; results table with n per class.
- Pipeline tested end-to-end on **simulated** labelled data (clearly labelled); the real report is generated only
  when real sessions exist.

---

### M13 — RL environment (gated: start only after M8 passes AND the user approves)
**Deps:** gymnasium, stable-baselines3, torch (install from the official PyTorch CUDA index matching the driver;
verify `torch.cuda.is_available()`).
- `rl/env.py`: Gymnasium env wrapping the M8 simulator. Observation = stroke features; action = bounded deltas of
  the M8 parameter vector (episode = K refinement steps) — or residual joint targets in MuJoCo (M8b) if approved.
  Reward = the **same versioned objective** as M8, including penalties.
- `rl/train.py`: PPO/SAC, ≥ 5 seeds, logged evaluation-call budget.
- `experiments/E13_rl_vs_bo/`: RL vs Optuna vs random search at **equal simulator-evaluation budget**, mean ± SD
  over seeds. "RL did not outperform BO" is a valid, reportable outcome.

---

### M14 — Experiment harness and Finale evidence pack
- `experiments/E00…E14/protocol.md` templates (purpose, setup, n, procedure, metrics, analysis script, threats to
  validity) for: E0 link quality, E1 static orientation/drift, E2 saturation, E3 rig joint angle vs encoder
  (camera / IMU / fused), E4 goniometer, E5 human camera-vs-IMU agreement, E6 sync residual, E7 glass-to-glass
  latency, E8 shadow positional error vs independent reference, E9 simulator validity (predicted vs measured
  shuttle speed/landing), E10 classification, E11 speed prediction, E12 optimizer comparison, E13 RL vs BO,
  E14 real-athlete pilot (A-B-A).
- `scripts/make_report.py`: collects every `experiments/*/results/*.json` into one HTML/Markdown evidence report
  with n, mean ± SD, median, p95, max, 95% CI — and marks each result REAL or SIMULATED.
- Demo hardening checklist: live → replay → pre-rendered video fallbacks; battery and radio checks.

---

## 8. References to cite in code/docs where used
- Bosch Sensortec community: BNO055 fusion mode uses fixed 4 g / 2000 dps ranges.
- "A Correlational Analysis of Shuttlecock Speed Kinematic Determinants in the Badminton Jump Smash",
  Applied Sciences (MDPI) 2020, 10(4):1248 — jump smash: racket head speed 56.3 ± 4.0 m/s,
  shuttle speed 89.6 ± 5.3 m/s (ratio ≈ 1.59, elite players; use only as a configurable default for `k_impact`),
  acceleration phase 38.1 ± 5.2 ms.
- arXiv 2601.01412 — shuttlecock terminal velocity ≈ 6.5 m/s (feather), quadratic drag.
- Seel, Raisch & Schauer (2014), IMU-based joint angle measurement for gait analysis — hinge-axis estimation.
- Wu et al. (2005), ISB recommendations, upper extremity joint coordinate systems.
- de Leva (1996), adjustments to Zatsiorsky–Seluyanov segment inertia parameters.
- MediaPipe Pose Landmarker (Tasks API) documentation — model files and running modes.

## 9. Glossary
**Digital Twin:** athlete-specific kinematic model (measured segment lengths, joint limits, racket) plus the
athlete's recorded motion distribution. **Motion Shadow:** the twin driven live by reconstructed motion.
**Candidate:** a parameterised variation of a recorded stroke evaluated in simulation. **REAL / SIMULATED:**
every result is tagged with one of these, always.
