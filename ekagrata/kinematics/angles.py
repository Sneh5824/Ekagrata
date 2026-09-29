"""Segment frames and joint angles (SPEC §3.1, §3.2). Right arm; angles in rad.

Segment frames (ISB style): +Y along the long axis pointing PROXIMALLY, +X anterior at the reference pose,
+Z = X x Y (lateral for the right arm). Reference pose: standing, arm hanging, palm facing the thigh.

ONE definition, TWO inputs: every derived angle is computed from joint quaternions / segment axes by the
functions below; the camera path (landmarks) and the IMU path (q_world_seg per segment, M7) both call them.

Shoulder (ISB Y-X-Y):  q_torso_upper_arm = Ry(plane) * Rx(-elev) * Ry(rot)
  elev  in [0, pi]  : 0 = arm hanging;   plane: 0 = pure abduction, +pi/2 = forward flexion;
  rot               : humeral axial rotation, NaN within `singular` of elev 0 or pi.
Elbow:  elbow_flex = angle between upper-arm and forearm long axes (0 = straight);
        forearm_pron = twist of q_upper_arm_forearm about forearm Y (not observable from the camera -> NaN).
Wrist:  swing-twist of q_forearm_hand about Y: wrist_flex = swing rotation-vector Z component,
        wrist_dev = X component, racket_twist = twist angle.
"""

import numpy as np
import pandas as pd

from ekagrata.core import quat
from ekagrata.vision.landmark_map import LM

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
ANGLES = ["shoulder_elev", "shoulder_plane", "shoulder_rot", "elbow_flex", "forearm_pron",
          "wrist_flex", "wrist_dev", "racket_twist"]
JOINT_QUATS = ["q_world_torso", "q_torso_upper_arm", "q_upper_arm_forearm", "q_forearm_hand"]
QUALITY = ["shoulder_quality", "elbow_quality", "wrist_quality"]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, np.nan)


def frame_from_y_and_hint(y, hint, hint_axis: str) -> np.ndarray:
    """q_world_seg from the segment +Y direction and a hint for +Z (hint_axis='z') or +X (hint_axis='x'),
    Gram-Schmidt orthogonalised against Y. NaN where inputs are NaN or degenerate."""
    Y = _unit(y)
    h = np.asarray(hint, dtype=np.float64)
    h = _unit(h - np.sum(h * Y, axis=-1, keepdims=True) * Y)
    if hint_axis == "z":
        Z = h
        X = np.cross(Y, Z)
    elif hint_axis == "x":
        X = h
        Z = np.cross(X, Y)
    else:
        raise ValueError("hint_axis must be 'x' or 'z'")
    return quat.from_matrix(np.stack([X, Y, Z], axis=-1))


# --- Shoulder ------------------------------------------------------------------------------------


def shoulder_angles_from_axis(y_h_torso) -> tuple[np.ndarray, np.ndarray]:
    """(elev, plane) from the humerus +Y axis expressed in the torso frame (only the axis is needed)."""
    y = _unit(y_h_torso)
    elev = np.arccos(np.clip(y[..., 1], -1.0, 1.0))
    plane = np.arctan2(-y[..., 0], -y[..., 2])
    return elev, plane


def shoulder_angles(q_torso_upper_arm, singular_rad: float = np.deg2rad(10.0)):
    """(elev, plane, rot) from the full shoulder quaternion (ISB Y-X-Y decomposition)."""
    q = np.asarray(q_torso_upper_arm, dtype=np.float64)
    elev, plane = shoulder_angles_from_axis(quat.rotate_vector(q, EY))
    q_plane = quat.from_rotvec(plane[..., None] * EY)
    q_elev = quat.from_rotvec(-elev[..., None] * EX)
    with np.errstate(invalid="ignore"):
        q_rest = quat.multiply(quat.conj(quat.multiply(q_plane, q_elev)), q)
    ok = np.all(np.isfinite(q_rest), axis=-1)
    rot = np.full(elev.shape, np.nan)
    if np.any(ok):
        rot[ok] = quat.twist_angle(q_rest[ok], EY)
    singular = (elev < singular_rad) | (elev > np.pi - singular_rad)
    rot = np.where(singular, np.nan, rot)
    return elev, plane, rot


# --- Elbow ---------------------------------------------------------------------------------------


def elbow_flex_from_axes(y_upper_arm, y_forearm) -> np.ndarray:
    """Angle between the upper-arm and forearm +Y (proximal) axes, in any common frame. 0 = straight."""
    a, b = _unit(y_upper_arm), _unit(y_forearm)
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def elbow_angles(q_upper_arm_forearm) -> tuple[np.ndarray, np.ndarray]:
    """(elbow_flex, forearm_pron) from the full elbow quaternion."""
    q = np.asarray(q_upper_arm_forearm, dtype=np.float64)
    flex = elbow_flex_from_axes(np.broadcast_to(EY, q.shape[:-1] + (3,)), quat.rotate_vector(q, EY))
    ok = np.all(np.isfinite(q), axis=-1)
    pron = np.full(flex.shape, np.nan)
    if np.any(ok):
        pron[ok] = quat.twist_angle(q[ok], EY)
    return flex, pron


# --- Wrist ---------------------------------------------------------------------------------------


def wrist_angles(q_forearm_hand) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wrist_flex, wrist_dev, racket_twist): swing rotation vector Z and X components, and twist about Y."""
    q = np.asarray(q_forearm_hand, dtype=np.float64)
    shape = q.shape[:-1]
    flex, dev, twist = (np.full(shape, np.nan) for _ in range(3))
    ok = np.all(np.isfinite(q), axis=-1)
    if np.any(ok):
        swing, _ = quat.swing_twist(q[ok], EY)
        rv = quat.to_rotvec(swing)
        flex[ok], dev[ok] = rv[..., 2], rv[..., 0]
        twist[ok] = quat.twist_angle(q[ok], EY)
    return flex, dev, twist


# --- Camera path ---------------------------------------------------------------------------------


def camera_segment_frames(lm_world, min_flex_rad: float) -> dict[str, np.ndarray]:
    """Segment orientations q_world_seg from world landmarks (N, 33, 3) of a right-handed athlete.

    torso: Y = mid-shoulder - mid-hip, Z hint = right - left shoulder.
    upper_arm: Y = shoulder - elbow, Z = (wrist - elbow) x Y; NaN if elbow_flex < min_flex (plane undefined).
    forearm: Y = elbow - wrist; X/Z ASSUMED from the upper-arm frame carried through the elbow with zero
        pronation (pronation is unobservable, SPEC §1.4) -> only used for low-quality wrist angles.
    hand: Y = wrist - mid(index, pinky), X hint = index - pinky (anterior at the reference pose).
    """
    p = np.asarray(lm_world, dtype=np.float64)
    sh, el, wr = p[:, LM["r_shoulder"]], p[:, LM["r_elbow"]], p[:, LM["r_wrist"]]
    mid_hip = 0.5 * (p[:, LM["l_hip"]] + p[:, LM["r_hip"]])
    mid_sh = 0.5 * (p[:, LM["l_shoulder"]] + sh)
    q_torso = frame_from_y_and_hint(mid_sh - mid_hip, sh - p[:, LM["l_shoulder"]], "z")

    y_ua, y_fa = sh - el, el - wr
    flex = elbow_flex_from_axes(y_ua, y_fa)
    q_ua = frame_from_y_and_hint(y_ua, np.cross(wr - el, y_ua), "z")
    q_ua = np.where((flex >= min_flex_rad)[:, None], q_ua, np.nan)

    q_fa = quat.multiply(quat.from_two_vectors(y_ua, y_fa), q_ua)

    idx, pky = p[:, LM["r_index"]], p[:, LM["r_pinky"]]
    q_hand = frame_from_y_and_hint(wr - 0.5 * (idx + pky), idx - pky, "x")
    return {"torso": q_torso, "upper_arm": q_ua, "forearm": q_fa, "hand": q_hand,
            "y_upper_arm": _unit(y_ua), "y_forearm": _unit(y_fa)}


def _quality(ok, level: str) -> np.ndarray:
    return np.where(ok, level, "none")


def joint_angles_from_camera(lm_world, min_flex_rad: float = np.deg2rad(15.0),
                             singular_rad: float = np.deg2rad(10.0)) -> pd.DataFrame:
    """Joint quaternions, derived angles and quality flags from world landmarks (N, 33, 3).
    Observability per SPEC §3.2: forearm_pron and racket_twist are NaN; wrist angles are quality 'low';
    elbow and wrist quaternions are NaN (their twist is not observable)."""
    f = camera_segment_frames(lm_world, min_flex_rad)
    n = f["torso"].shape[0]
    q_t_ua = quat.relative(f["torso"], f["upper_arm"])

    y_h_torso = quat.rotate_vector(quat.conj(f["torso"]), f["y_upper_arm"])
    elev, plane = shoulder_angles_from_axis(y_h_torso)
    _, _, rot = shoulder_angles(q_t_ua, singular_rad)  # NaN where the upper-arm frame is undefined

    flex = elbow_flex_from_axes(f["y_upper_arm"], f["y_forearm"])
    w_flex, w_dev, _ = wrist_angles(quat.relative(f["forearm"], f["hand"]))

    out = {
        "shoulder_elev": elev, "shoulder_plane": plane, "shoulder_rot": rot,
        "elbow_flex": flex, "forearm_pron": np.full(n, np.nan),
        "wrist_flex": w_flex, "wrist_dev": w_dev, "racket_twist": np.full(n, np.nan),
    }
    quats = {"q_world_torso": f["torso"], "q_torso_upper_arm": q_t_ua,
             "q_upper_arm_forearm": np.full((n, 4), np.nan), "q_forearm_hand": np.full((n, 4), np.nan)}
    df = _frame(out, quats)
    df["shoulder_quality"] = _quality(np.isfinite(elev), "ok")
    df["elbow_quality"] = _quality(np.isfinite(flex), "ok")
    df["wrist_quality"] = _quality(np.isfinite(w_flex), "low")
    return df


# --- Quaternion (IMU) path -----------------------------------------------------------------------


def joint_angles_from_quats(q_world: dict[str, np.ndarray],
                            singular_rad: float = np.deg2rad(10.0)) -> pd.DataFrame:
    """Same columns as the camera path, from q_world_seg (N, 4) for 'torso', 'upper_arm', 'forearm' and
    'racket' (or 'hand'). All DoFs are observable here, so every angle is computed."""
    distal = q_world["racket"] if "racket" in q_world else q_world["hand"]
    q_t_ua = quat.relative(q_world["torso"], q_world["upper_arm"])
    q_ua_fa = quat.relative(q_world["upper_arm"], q_world["forearm"])
    q_fa_h = quat.relative(q_world["forearm"], distal)
    elev, plane, rot = shoulder_angles(q_t_ua, singular_rad)
    flex, pron = elbow_angles(q_ua_fa)
    w_flex, w_dev, twist = wrist_angles(q_fa_h)
    out = {
        "shoulder_elev": elev, "shoulder_plane": plane, "shoulder_rot": rot,
        "elbow_flex": flex, "forearm_pron": pron,
        "wrist_flex": w_flex, "wrist_dev": w_dev, "racket_twist": twist,
    }
    quats = {"q_world_torso": q_world["torso"], "q_torso_upper_arm": q_t_ua,
             "q_upper_arm_forearm": q_ua_fa, "q_forearm_hand": q_fa_h}
    df = _frame(out, quats)
    df["shoulder_quality"] = _quality(np.isfinite(elev), "ok")
    df["elbow_quality"] = _quality(np.isfinite(flex), "ok")
    df["wrist_quality"] = _quality(np.isfinite(w_flex), "ok")
    return df


def _frame(angles: dict, quats: dict) -> pd.DataFrame:
    cols = dict(angles)
    for name, q in quats.items():
        q = np.asarray(q, dtype=np.float64)
        for i, c in enumerate("wxyz"):
            cols[f"{name}_{c}"] = q[:, i]
    return pd.DataFrame(cols)


def interior_angle(a, b, c) -> np.ndarray:
    """Angle ABC at vertex B (rad), 3-D. For the elbow: elbow_flex = pi - interior_angle(S, E, W)."""
    return elbow_flex_from_axes(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))
