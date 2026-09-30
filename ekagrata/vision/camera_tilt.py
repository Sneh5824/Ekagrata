"""Camera tilt DIAGNOSTIC from the torso direction (check_axes.py rest step). Nothing here is applied to data.

MediaPipe world axes follow the camera, so EKAGRATA "Z up" (via mp_to_world) currently means camera-up, not
gravity-up. ASSUMPTION: the torso (mid-hip -> mid-shoulder) is vertical when standing relaxed. Under that
assumption the torso direction t, expressed in the camera-aligned EKAGRATA frame (X toward the camera,
Y = image right = the facing person's left, Z camera-up), reveals the camera tilt:
  - pitch (positive = camera looks down):  t = (sin p, 0, cos p)  ->  p = atan2(t_x, t_z)
  - roll (positive = camera rotated clockwise, seen from behind the camera): true up then appears tilted
    toward image left (-Y)  ->  r = atan2(-t_y, t_z)
Any real torso lean adds directly to these numbers, so they are estimates, not measurements of the camera.
"""

import numpy as np

from ekagrata.vision.landmark_map import LM


def torso_vector(world_points) -> np.ndarray:
    """mid-hip -> mid-shoulder vector (3,) from one frame of EKAGRATA-world landmarks (33, 3)."""
    p = np.asarray(world_points, dtype=np.float64)
    mid_hip = (p[LM["l_hip"]] + p[LM["r_hip"]]) / 2
    mid_shoulder = (p[LM["l_shoulder"]] + p[LM["r_shoulder"]]) / 2
    return mid_shoulder - mid_hip


def estimate_camera_tilt(torso_vectors) -> tuple[float, float, int]:
    """(pitch_deg, roll_deg, n) from the per-axis median of the finite torso vectors (N, 3) of a still pose.
    NaN pitch/roll if there are no finite vectors."""
    t = np.asarray(torso_vectors, dtype=np.float64).reshape(-1, 3)
    t = t[np.all(np.isfinite(t), axis=1)]
    if t.shape[0] == 0:
        return float("nan"), float("nan"), 0
    m = np.median(t, axis=0)
    pitch = np.degrees(np.arctan2(m[0], m[2]))
    roll = np.degrees(np.arctan2(-m[1], m[2]))
    return float(pitch), float(roll), int(t.shape[0])
