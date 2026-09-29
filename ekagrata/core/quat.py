"""Quaternion math. The ONLY module that talks to scipy.spatial.transform.Rotation.

Convention: Hamilton quaternions stored as numpy arrays [w, x, y, z] (scalar-first).
`q_a_b` rotates a vector expressed in frame b into frame a: v_a = q_a_b * v_b * conj(q_a_b).
Composition: q_a_c = q_a_b * q_b_c.

Every function accepts a single quaternion of shape (4,) or a stack of shape (N, 4)
(generally (..., 4)) and broadcasts like numpy.
"""

import numpy as np
from scipy.spatial.transform import Rotation

_EPS = 1e-12


def _as_array(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion last dimension must be 4, got shape {q.shape}")
    return q


def normalize(q) -> np.ndarray:
    """Return unit quaternion(s). Raises ValueError on a (near-)zero quaternion."""
    q = _as_array(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < _EPS):
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def conj(q) -> np.ndarray:
    """Quaternion conjugate [w, -x, -y, -z] (the inverse for unit quaternions)."""
    q = _as_array(q)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def multiply(q1, q2) -> np.ndarray:
    """Hamilton product q1 * q2."""
    q1 = _as_array(q1)
    q2 = _as_array(q2)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def rotate_vector(q_a_b, v_b) -> np.ndarray:
    """Rotate vector(s) v_b (shape (..., 3)) from frame b into frame a using unit q_a_b."""
    q = _as_array(q_a_b)
    v = np.asarray(v_b, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"vector last dimension must be 3, got shape {v.shape}")
    w = q[..., :1]
    u = q[..., 1:]
    # v' = v + 2 w (u x v) + 2 u x (u x v), equivalent to q * [0, v] * conj(q) for unit q.
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def from_scipy(rot: Rotation) -> np.ndarray:
    """Convert a SciPy Rotation to [w, x, y, z]."""
    return np.roll(rot.as_quat(), 1, axis=-1)


def to_scipy(q) -> Rotation:
    """Convert [w, x, y, z] to a SciPy Rotation (SciPy uses [x, y, z, w])."""
    return Rotation.from_quat(np.roll(_as_array(q), -1, axis=-1))


def from_matrix(R) -> np.ndarray:
    """Rotation matrix/matrices (..., 3, 3) (columns = child axes in parent frame) to unit quaternion(s).
    Rows containing NaN give NaN quaternions."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got shape {R.shape}")
    flat = R.reshape(-1, 3, 3)
    out = np.full((flat.shape[0], 4), np.nan)
    ok = np.all(np.isfinite(flat), axis=(1, 2))
    if np.any(ok):
        out[ok] = from_scipy(Rotation.from_matrix(flat[ok]))
    return out.reshape(R.shape[:-2] + (4,))


def to_matrix(q) -> np.ndarray:
    """Unit quaternion(s) to rotation matrix/matrices (..., 3, 3). NaN quaternions give NaN matrices."""
    q = _as_array(q)
    flat = q.reshape(-1, 4)
    out = np.full((flat.shape[0], 3, 3), np.nan)
    ok = np.all(np.isfinite(flat), axis=1)
    if np.any(ok):
        out[ok] = to_scipy(flat[ok]).as_matrix()
    return out.reshape(q.shape[:-1] + (3, 3))


def from_two_vectors(a, b) -> np.ndarray:
    """Smallest rotation q with rotate_vector(q, â) = b̂ (no twist about the vectors). For antiparallel inputs
    a 180 deg rotation about an arbitrary axis perpendicular to a is returned."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    a, b = np.broadcast_arrays(a, b)
    w = 1.0 + np.sum(a * b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    anti = (w < 1e-9)[..., 0]
    if np.any(anti):
        # Perpendicular axis: cross with X, or with Y when a is (nearly) along X.
        helper = np.where(np.abs(a[anti][:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        v = v.copy()
        w = w.copy()
        v[anti] = np.cross(a[anti], helper)
        w[anti] = 0.0
    q = np.concatenate([w, v], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_rotvec(rotvec) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of shape (..., 3) to unit quaternion(s)."""
    r = np.asarray(rotvec, dtype=np.float64)
    if r.shape[-1] != 3:
        raise ValueError(f"rotvec last dimension must be 3, got shape {r.shape}")
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    # sin(angle/2)/angle, well-defined at angle = 0 (np.sinc(x) = sin(pi x)/(pi x)).
    scale = 0.5 * np.sinc(angle / (2.0 * np.pi))
    return np.concatenate([np.cos(angle / 2.0), r * scale], axis=-1)


def to_rotvec(q) -> np.ndarray:
    """Unit quaternion(s) to rotation vector(s) with angle in [0, pi]."""
    q = normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)  # pick the w >= 0 representative
    w = q[..., :1]
    u = q[..., 1:]
    s = np.linalg.norm(u, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, w)
    small = s < 1e-9
    scale = np.where(small, 2.0 / np.where(small, w, 1.0), angle / np.where(small, 1.0, s))
    return u * scale


def relative(q_world_parent, q_world_child) -> np.ndarray:
    """Joint rotation q_parent_child = conj(q_world_parent) * q_world_child."""
    return multiply(conj(q_world_parent), q_world_child)


def angle_between(q1, q2) -> np.ndarray:
    """Rotation angle (rad, in [0, pi]) between orientations q1 and q2; q and -q are the same rotation."""
    d = multiply(conj(normalize(q1)), normalize(q2))
    return 2.0 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def swing_twist(q, axis) -> tuple[np.ndarray, np.ndarray]:
    """Decompose q = q_swing * q_twist, where q_twist rotates about `axis` and q_swing about an axis
    perpendicular to it.

    `axis` is a (3,) (or broadcastable (..., 3)) vector expressed in the child frame of q, e.g. the forearm +Y
    axis for q_upper_arm_forearm. The twist is canonicalised to w >= 0. When the twist is undefined (a 180 deg
    rotation about an axis perpendicular to `axis`), the twist is the identity and the swing equals q.
    """
    q = normalize(q)
    a = np.asarray(axis, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"axis last dimension must be 3, got shape {a.shape}")
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(a_norm < _EPS):
        raise ValueError("twist axis must be non-zero")
    a = a / a_norm
    proj = np.sum(q[..., 1:] * a, axis=-1, keepdims=True)
    lead = np.broadcast_shapes(q.shape[:-1], a.shape[:-1])
    twist = np.concatenate(
        [np.broadcast_to(q[..., :1], lead + (1,)), np.broadcast_to(proj * a, lead + (3,))], axis=-1
    )
    n = np.linalg.norm(twist, axis=-1, keepdims=True)
    singular = n < _EPS
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    twist = np.where(singular, identity, twist / np.where(singular, 1.0, n))
    twist = np.where(twist[..., :1] < 0.0, -twist, twist)
    swing = multiply(q, conj(twist))
    return swing, twist


def twist_angle(q, axis) -> np.ndarray:
    """Signed rotation angle (rad, in (-pi, pi]) of the twist part of q about `axis` (right-hand rule)."""
    _, twist = swing_twist(q, axis)
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(np.sum(twist[..., 1:] * a, axis=-1), twist[..., 0])
    return np.where(angle <= -np.pi, angle + 2.0 * np.pi, angle)


def slerp(q0, q1, t) -> np.ndarray:
    """Spherical linear interpolation along the shortest arc; t in [0, 1], scalar or array broadcastable
    against the leading dimensions of q0/q1."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    safe_sin = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    w1 = np.where(near, t, np.sin(t * theta) / safe_sin)
    return normalize(w0 * q0 + w1 * q1)
