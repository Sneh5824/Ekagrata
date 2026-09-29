import numpy as np
import pytest

from ekagrata.core import quat
from ekagrata.kinematics.angles import (
    ANGLES,
    elbow_angles,
    interior_angle,
    joint_angles_from_camera,
    joint_angles_from_quats,
    shoulder_angles,
    wrist_angles,
)
from ekagrata.vision.landmark_map import LM

EX, EY, EZ = np.eye(3)
L_UA, L_FA, L_HAND = 0.30, 0.25, 0.08


def rx(a):
    return quat.from_rotvec(a * EX)


def ry(a):
    return quat.from_rotvec(a * EY)


def rz(a):
    return quat.from_rotvec(a * EZ)


def skeleton_from_frames(q_torso, q_ua, q_fa, q_hand, origin=(0.0, 0.0, 0.0)):
    """World landmarks (33, 3) consistent with the given segment orientations (right arm)."""
    ax = lambda q, e: quat.rotate_vector(q, e)  # noqa: E731
    origin = np.asarray(origin, dtype=float)
    p = np.full((33, 3), np.nan)
    Yt, Zt = ax(q_torso, EY), ax(q_torso, EZ)
    p[LM["r_hip"]], p[LM["l_hip"]] = origin + 0.1 * Zt, origin - 0.1 * Zt
    top = origin + 0.5 * Yt
    p[LM["r_shoulder"]], p[LM["l_shoulder"]] = top + 0.2 * Zt, top - 0.2 * Zt
    p[LM["r_elbow"]] = p[LM["r_shoulder"]] - L_UA * ax(q_ua, EY)
    p[LM["r_wrist"]] = p[LM["r_elbow"]] - L_FA * ax(q_fa, EY)
    mid = p[LM["r_wrist"]] - L_HAND * ax(q_hand, EY)
    p[LM["r_index"]] = mid + 0.02 * ax(q_hand, EX)
    p[LM["r_pinky"]] = mid - 0.02 * ax(q_hand, EX)
    return p


def arm_pose(q_torso, plane, elev, rot, flex, wrist_q=None):
    """Segment orientations from joint angles (zero pronation), expressed in world via q_torso."""
    q_t_ua = quat.multiply(quat.multiply(ry(plane), rx(-elev)), ry(rot))
    q_ua = quat.multiply(q_torso, q_t_ua)
    q_fa = quat.multiply(q_ua, rz(flex))
    q_hand = quat.multiply(q_fa, np.array([1.0, 0, 0, 0]) if wrist_q is None else wrist_q)
    return q_ua, q_fa, q_hand


def camera_angles(landmarks):
    return joint_angles_from_camera(np.asarray(landmarks)[None] if np.ndim(landmarks) == 2 else landmarks)


def simple_skeleton(d, f):
    """Upright torso facing +X_world, humerus distal direction d and forearm distal direction f (world)."""
    q_torso = quat.from_matrix(np.stack([EX, EZ, -EY], axis=-1))  # torso X=+Xw, Y=+Zw (up), Z=-Yw (right)
    p = skeleton_from_frames(q_torso, np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]),
                             np.array([1.0, 0, 0, 0]))
    p[LM["r_elbow"]] = p[LM["r_shoulder"]] + L_UA * np.asarray(d, float)
    p[LM["r_wrist"]] = p[LM["r_elbow"]] + L_FA * np.asarray(f, float)
    p[LM["r_index"]] = p[LM["r_wrist"]] + 0.08 * np.asarray(f, float) + [0.02, 0, 0]
    p[LM["r_pinky"]] = p[LM["r_wrist"]] + 0.08 * np.asarray(f, float) - [0.02, 0, 0]
    return p


DOWN, RIGHT, FORWARD, UP = np.array([0, 0, -1.0]), np.array([0, -1.0, 0]), EX, EZ


# --- SPEC M2 synthetic skeletons -------------------------------------------------------------------


def test_arm_down_elev_zero_elbow_straight():
    df = camera_angles(simple_skeleton(DOWN, DOWN))
    assert df.shoulder_elev[0] == pytest.approx(0.0, abs=1e-9)
    assert df.elbow_flex[0] == pytest.approx(0.0, abs=1e-9)


def test_horizontal_abduction():
    df = camera_angles(simple_skeleton(RIGHT, RIGHT))
    assert df.shoulder_elev[0] == pytest.approx(np.pi / 2)
    assert df.shoulder_plane[0] == pytest.approx(0.0, abs=1e-9)


def test_forward_flexion():
    df = camera_angles(simple_skeleton(FORWARD, FORWARD))
    assert df.shoulder_elev[0] == pytest.approx(np.pi / 2)
    assert df.shoulder_plane[0] == pytest.approx(np.pi / 2)


def test_full_elevation_overhead():
    df = camera_angles(simple_skeleton(UP, UP))
    assert df.shoulder_elev[0] == pytest.approx(np.pi)


def test_elbow_90_bend():
    df = camera_angles(simple_skeleton(DOWN, FORWARD))
    assert df.elbow_flex[0] == pytest.approx(np.pi / 2)
    assert df.shoulder_elev[0] == pytest.approx(0.0, abs=1e-9)


def test_proposal_2d_example_is_135_interior_45_flex():
    # SPEC §1.3: the PDF's 2-D elbow example; interior angle 135.0 deg -> elbow_flex 45 deg.
    a, b, c = (np.array([x, y, 0.0]) for x, y in ((0.20, 1.50), (0.35, 1.25), (0.55, 1.20)))
    assert np.rad2deg(interior_angle(a, b, c)) == pytest.approx(135.0, abs=1e-6)
    assert np.rad2deg(np.pi - interior_angle(a, b, c)) == pytest.approx(45.0, abs=1e-6)


# --- ISB decomposition and one-definition-two-inputs ----------------------------------------------


@pytest.mark.parametrize("plane,elev,rot", [(0.4, 1.0, 0.6), (-1.2, 2.0, -0.9), (2.5, 0.5, 1.4)])
def test_shoulder_yxy_recovers_known_angles(plane, elev, rot):
    q = quat.multiply(quat.multiply(ry(plane), rx(-elev)), ry(rot))
    e, p, r = shoulder_angles(q)
    assert (e, p, r) == pytest.approx((elev, plane, rot), abs=1e-9)


def test_shoulder_rot_nan_near_singularity():
    q = quat.multiply(quat.multiply(ry(0.3), rx(-np.deg2rad(5))), ry(0.5))
    _, _, r = shoulder_angles(q, singular_rad=np.deg2rad(10))
    assert np.isnan(r)


def test_elbow_and_wrist_quaternion_known_answers():
    flex, pron = elbow_angles(quat.multiply(rz(1.0), ry(0.4)))
    assert (flex, pron) == pytest.approx((1.0, 0.4))
    assert wrist_angles(rz(0.3)) == pytest.approx((0.3, 0.0, 0.0), abs=1e-12)
    assert wrist_angles(rx(0.2)) == pytest.approx((0.0, 0.2, 0.0), abs=1e-12)
    assert wrist_angles(ry(0.5)) == pytest.approx((0.0, 0.0, 0.5), abs=1e-12)


def test_camera_and_quaternion_paths_agree():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q_t = quat.normalize(rng.normal(size=4))
        plane, rot = rng.uniform(-2.5, 2.5, size=2)
        elev = rng.uniform(0.4, 2.7)
        flex = rng.uniform(0.4, 2.3)
        wrist_q = quat.multiply(rz(rng.uniform(-0.6, 0.6)), rx(rng.uniform(-0.3, 0.3)))  # swing only
        q_ua, q_fa, q_h = arm_pose(q_t, plane, elev, rot, flex, wrist_q)
        cam = camera_angles(skeleton_from_frames(q_t, q_ua, q_fa, q_h))
        imu = joint_angles_from_quats({"torso": q_t[None], "upper_arm": q_ua[None], "forearm": q_fa[None],
                                       "hand": q_h[None]})
        for name in ("shoulder_elev", "shoulder_plane", "shoulder_rot", "elbow_flex", "wrist_flex",
                     "wrist_dev"):
            assert cam[name][0] == pytest.approx(imu[name][0], abs=1e-9), name
        assert imu.shoulder_elev[0] == pytest.approx(elev)
        assert imu.shoulder_rot[0] == pytest.approx(rot)
        assert imu.elbow_flex[0] == pytest.approx(flex)
        assert imu.forearm_pron[0] == pytest.approx(0.0, abs=1e-9)


def test_global_rotation_invariance():
    rng = np.random.default_rng(11)
    q_t = quat.normalize(rng.normal(size=4))
    base = skeleton_from_frames(q_t, *arm_pose(q_t, 0.7, 1.3, -0.4, 1.1, rz(0.2)))
    ref = camera_angles(base)
    for _ in range(10):
        g = quat.normalize(rng.normal(size=4))
        moved = quat.rotate_vector(g, base) + rng.normal(size=3)
        df = camera_angles(moved)
        for name in ANGLES:
            np.testing.assert_allclose(df[name].to_numpy(), ref[name].to_numpy(), atol=1e-9, equal_nan=True)


# --- Observability (SPEC §3.2) and NaN propagation -------------------------------------------------


def test_camera_observability_flags():
    q_t = quat.from_matrix(np.stack([EX, EZ, -EY], axis=-1))
    df = camera_angles(skeleton_from_frames(q_t, *arm_pose(q_t, 0.5, 1.0, 0.3, 1.2)))
    assert np.isnan(df.forearm_pron[0]) and np.isnan(df.racket_twist[0])
    assert df.wrist_quality[0] == "low" and df.shoulder_quality[0] == "ok"
    assert np.all(np.isnan(df[[f"q_upper_arm_forearm_{c}" for c in "wxyz"]].to_numpy()))
    assert np.all(np.isfinite(df[[f"q_torso_upper_arm_{c}" for c in "wxyz"]].to_numpy()))


def test_straight_elbow_leaves_rot_nan_but_elev_plane_defined():
    q_t = quat.from_matrix(np.stack([EX, EZ, -EY], axis=-1))
    df = camera_angles(skeleton_from_frames(q_t, *arm_pose(q_t, 0.5, 1.0, 0.3, np.deg2rad(10))))
    assert np.isnan(df.shoulder_rot[0])
    assert df.shoulder_elev[0] == pytest.approx(1.0)
    assert df.shoulder_plane[0] == pytest.approx(0.5)
    assert np.all(np.isnan(df[[f"q_torso_upper_arm_{c}" for c in "wxyz"]].to_numpy()))


def test_missing_landmark_propagates_nan():
    p = simple_skeleton(DOWN, FORWARD)
    p[LM["r_wrist"]] = np.nan
    df = camera_angles(p)
    assert np.isnan(df.elbow_flex[0]) and df.elbow_quality[0] == "none"
    assert np.isfinite(df.shoulder_elev[0])
    p2 = simple_skeleton(DOWN, FORWARD)
    p2[LM["l_hip"]] = np.nan
    df2 = camera_angles(p2)
    assert np.isnan(df2.shoulder_elev[0]) and df2.shoulder_quality[0] == "none"
    assert np.isfinite(df2.elbow_flex[0])


def test_batched_frames():
    frames = np.stack([simple_skeleton(DOWN, DOWN), simple_skeleton(FORWARD, FORWARD),
                       simple_skeleton(DOWN, UP)])
    df = joint_angles_from_camera(frames)
    assert len(df) == 3
    np.testing.assert_allclose(df.shoulder_elev, [0, np.pi / 2, 0], atol=1e-9)
    np.testing.assert_allclose(df.elbow_flex, [0, 0, np.pi], atol=1e-9)
