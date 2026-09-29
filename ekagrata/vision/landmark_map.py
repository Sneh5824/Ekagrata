"""The ONE place where MediaPipe world-landmark axes are converted to the EKAGRATA world frame.

MediaPipe world landmarks: metres, origin at the hip midpoint, x = image right, y = image down,
z = away from the camera (depth). EKAGRATA world: right-handed, Z up (REP-103).
The mapping is a proper rotation `mp_to_world` (det = +1) from configs/camera.yaml, confirmed with
scripts/check_axes.py. Mirroring (det = -1) is rejected: left-handed athletes are mirrored explicitly
elsewhere.
"""

import numpy as np

# MediaPipe PoseLandmarker indices. Left/right are the PERSON's anatomical sides, not image sides.
LM = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24,
}
N_LANDMARKS = 33

# Default for an athlete facing the camera: X_w = -z_mp (toward camera), Y_w = +x_mp, Z_w = -y_mp (up).
FACING_CAMERA = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def validate_rotation(M, tol: float = 1e-6) -> np.ndarray:
    """Return M as a (3, 3) array if it is a proper rotation (orthonormal, det = +1); else ValueError."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (3, 3) or not np.all(np.isfinite(M)):
        raise ValueError(f"mp_to_world must be a finite 3x3 matrix, got shape {M.shape}")
    if not np.allclose(M @ M.T, np.eye(3), atol=tol):
        raise ValueError("mp_to_world must be orthonormal (M @ M.T == I)")
    det = float(np.linalg.det(M))
    if abs(det - 1.0) > tol:
        raise ValueError(f"mp_to_world must have det = +1 (no mirroring), got det = {det:+.6f}")
    return M


def mp_to_world(points_mp, M) -> np.ndarray:
    """Map MediaPipe world points (..., 3) to EKAGRATA world (..., 3), metres. NaN propagates."""
    M = validate_rotation(M)
    p = np.asarray(points_mp, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"points last dimension must be 3, got shape {p.shape}")
    return p @ M.T
