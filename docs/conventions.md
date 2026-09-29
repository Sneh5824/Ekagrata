# EKAGRATA conventions

These rules apply to every module and are enforced by tests.

## Units
SI internally: m, s, rad, rad/s, m/s². Degrees are for display only, and any such variable or column
name ends in `_deg`.

## Quaternions
- Hamilton convention, stored as numpy arrays **`[w, x, y, z]`** (scalar first).
- SciPy uses `[x, y, z, w]`. Convert **only** through `ekagrata/core/quat.py` (`from_scipy` / `to_scipy`).
  Never call `scipy.spatial.transform.Rotation.from_quat` anywhere else.

## Frame naming
- `q_a_b` rotates a vector expressed in frame b into frame a: `v_a = q_a_b ⊗ v_b ⊗ conj(q_a_b)`.
- Composition: `q_a_c = q_a_b ⊗ q_b_c` (inner frame names cancel).
- Joint rotation: `q_parent_child = conj(q_world_parent) ⊗ q_world_child` (`quat.relative`).

## Swing–twist decomposition
`quat.swing_twist(q, axis)` splits `q = q_swing ⊗ q_twist`: `q_twist` rotates about `axis` (a vector in the
**child** frame of `q`, e.g. forearm +Y for `q_upper_arm_forearm`), and `q_swing` rotates about an axis
perpendicular to it. `q_twist = normalize([w, (v·a) a])`, canonicalised to w ≥ 0, and `q_swing = q ⊗ conj(q_twist)`.
`quat.twist_angle(q, axis)` is the signed twist angle (right-hand rule) in (−π, π]. If the twist is undefined
(a 180° rotation about an axis perpendicular to `axis`), the twist is the identity. Used for `forearm_pron`
(elbow twist) and `racket_twist` / wrist swing (see SPEC §3.1).

Example: `q = rotX(25°) ⊗ rotY(70°)` with axis +Y gives `twist_angle = 70°` and a swing of 25° about X.

## World frame
Right-handed, Z up (ROS REP-103: x forward, y left, z up). MediaPipe axes are converted to this frame in
exactly one place: `ekagrata/vision/landmark_map.py`, using the `mp_to_world` rotation in `configs/camera.yaml`.

MediaPipe world landmarks: metres, origin at the hip midpoint, x = image right, y = image down, z = away from
the camera. Default mapping (athlete **facing** the camera): `X_w = −z_mp` (toward the camera),
`Y_w = +x_mp`, `Z_w = −y_mp` (up), i.e. `[[0,0,−1],[1,0,0],[0,−1,0]]` (det = +1). Confirm it live with
`scripts/check_axes.py`. Mirror matrices (det = −1) are rejected.

## Segment frames and joint angles (SPEC §3.1, `ekagrata/kinematics/angles.py`)
Every segment: +Y along the long axis pointing **proximally**, +X anterior at the reference pose, +Z = X × Y
(lateral for the right arm). Reference pose: standing, arm hanging, palm facing the thigh.

| Joint | Definition | Camera |
|---|---|---|
| shoulder | `q_torso_upper_arm = Ry(plane) ⊗ Rx(−elev) ⊗ Ry(rot)` (ISB Y-X-Y). `elev` ∈ [0, π], 0 = arm down; `plane` 0 = abduction, +π/2 = forward flexion | elev/plane need hips + shoulders + elbow; `rot` only when elbow flex ≥ 15°, NaN within 10° of elev 0/π |
| elbow | `elbow_flex` = angle between upper-arm and forearm +Y axes (0 = straight); `forearm_pron` = twist about forearm Y | flex observable; pronation **NaN** |
| wrist | swing–twist of `q_forearm_hand` about Y: `wrist_flex` = swing rotation-vector Z, `wrist_dev` = X, `racket_twist` = twist | flex/dev quality **low** (assumes zero pronation); twist **NaN** |

Camera-derived torso frame: `Y = mid-shoulder − mid-hip`, `Z` from left → right shoulder. The same angle
functions are used for camera landmarks and for IMU quaternions ("one definition, two inputs").
Example: a 3-point elbow interior angle of 135° (the original PDF's 2-D example) is `elbow_flex = 45°`.

## Time
- Integer nanoseconds, `int64`, names ending in `_ns` (or `_us` for ESP32 `micros()`).
- Host clock: `time.perf_counter_ns()` (`timebase.now_ns()`). Never use `time.time()` to measure intervals.
- ESP32 `micros()` wraps at 2³²; unwrap with `timebase.unwrap_u32`.
- Canonical timeline column: `t_sync_ns`.

## Segment / node names
`upper_arm`, `forearm`, `racket` (or `hand` if configured), `torso`.

## Data
- Raw session data lives under `data/sessions/<YYYY-MM-DD>_<session>_<athlete>/` and is append-only:
  never modify or delete it.
- Raw streams are CSV, derived tables are Parquet, metadata is `session.json`.
- Every IMU row carries `source` ∈ {`real`, `sim`, `replay`}.

## Worked example: elbow joint rotation
The upper-arm IMU reports `q_world_upper_arm = rotZ(90°)` (the athlete has turned 90° left). The
forearm IMU reports `q_world_forearm = rotZ(90°) ⊗ rotY(60°)` (the same turn, then 60° of elbow flexion
about the upper-arm y axis).

```
q_upper_arm_forearm = conj(q_world_upper_arm) ⊗ q_world_forearm
                    = conj(rotZ(90°)) ⊗ rotZ(90°) ⊗ rotY(60°)
                    = rotY(60°)
                    = [cos 30°, 0, sin 30°, 0] ≈ [0.866, 0, 0.5, 0]
```

The whole-body turn cancels and only the 60° elbow rotation remains. In code:

```python
import numpy as np
from ekagrata.core import quat

q_world_upper_arm = quat.from_rotvec([0.0, 0.0, np.pi / 2])
q_world_forearm = quat.multiply(q_world_upper_arm, quat.from_rotvec([0.0, np.pi / 3, 0.0]))
q_upper_arm_forearm = quat.relative(q_world_upper_arm, q_world_forearm)  # ≈ [0.866, 0, 0.5, 0]
elbow_angle_deg = np.rad2deg(quat.angle_between([1, 0, 0, 0], q_upper_arm_forearm))  # 60.0
```
