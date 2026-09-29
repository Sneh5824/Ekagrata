import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ekagrata.core import quat

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def rot_z(angle_rad: float) -> np.ndarray:
    return np.array([np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)])


def assert_same_rotation(q1, q2, atol=1e-9):
    """Equal up to the q / -q double cover."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    sign = np.sign(np.sum(q1 * q2, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    np.testing.assert_allclose(q1, sign * q2, atol=atol)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_quats(rng, n):
    return quat.normalize(rng.normal(size=(n, 4)))


# --- SciPy conversion and ordering -------------------------------------------------------------


def test_scipy_ordering_is_explicit():
    # SciPy identity is [x, y, z, w] = [0, 0, 0, 1]; ours is [w, x, y, z] = [1, 0, 0, 0].
    np.testing.assert_allclose(quat.from_scipy(Rotation.identity()), IDENTITY)
    q = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(quat.to_scipy(q).as_quat(), quat.normalize([0.2, 0.3, 0.4, 0.1]))


def test_scipy_round_trip_ours_to_scipy_to_ours(rng):
    q = random_quats(rng, 100)
    assert_same_rotation(quat.from_scipy(quat.to_scipy(q)), q)


def test_scipy_round_trip_scipy_to_ours_to_scipy(rng):
    r = Rotation.random(100, random_state=7)
    back = quat.to_scipy(quat.from_scipy(r))
    np.testing.assert_allclose(back.as_matrix(), r.as_matrix(), atol=1e-12)


# --- Known answers -----------------------------------------------------------------------------


def test_rot_z_90_maps_x_to_y():
    v = quat.rotate_vector(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)


def test_composition_q_a_c_equals_q_a_b_times_q_b_c(rng):
    q_a_b, q_b_c = random_quats(rng, 2)
    v_c = rng.normal(size=3)
    q_a_c = quat.multiply(q_a_b, q_b_c)
    expected = quat.rotate_vector(q_a_b, quat.rotate_vector(q_b_c, v_c))
    np.testing.assert_allclose(quat.rotate_vector(q_a_c, v_c), expected, atol=1e-12)


def test_rotate_vector_matches_sandwich_product(rng):
    q = random_quats(rng, 20)
    v = rng.normal(size=(20, 3))
    v_quat = np.concatenate([np.zeros((20, 1)), v], axis=1)
    sandwich = quat.multiply(quat.multiply(q, v_quat), quat.conj(q))
    np.testing.assert_allclose(quat.rotate_vector(q, v), sandwich[:, 1:], atol=1e-12)


def test_relative_identity_when_parent_equals_child(rng):
    q = random_quats(rng, 10)
    assert_same_rotation(quat.relative(q, q), np.tile(IDENTITY, (10, 1)))


def test_relative_recovers_30_deg_about_z(rng):
    q_world_parent = random_quats(rng, 1)[0]
    q_world_child = quat.multiply(q_world_parent, rot_z(np.deg2rad(30.0)))
    q_parent_child = quat.relative(q_world_parent, q_world_child)
    assert_same_rotation(q_parent_child, rot_z(np.deg2rad(30.0)))
    assert quat.angle_between(IDENTITY, q_parent_child) == pytest.approx(np.deg2rad(30.0))


# --- Double cover, angle, slerp ----------------------------------------------------------------


def test_angle_between_q_and_minus_q_is_zero(rng):
    q = random_quats(rng, 50)
    np.testing.assert_allclose(quat.angle_between(q, -q), 0.0, atol=1e-7)


def test_angle_between_known_value():
    assert quat.angle_between(IDENTITY, rot_z(np.deg2rad(170.0))) == pytest.approx(np.deg2rad(170.0))
    # 190 deg one way is 170 deg the other way.
    assert quat.angle_between(IDENTITY, rot_z(np.deg2rad(190.0))) == pytest.approx(np.deg2rad(170.0))


def test_slerp_endpoints_and_midpoint():
    q0 = IDENTITY
    q1 = rot_z(np.deg2rad(90.0))
    assert_same_rotation(quat.slerp(q0, q1, 0.0), q0)
    assert_same_rotation(quat.slerp(q0, q1, 1.0), q1)
    assert_same_rotation(quat.slerp(q0, q1, 0.5), rot_z(np.deg2rad(45.0)))


def test_slerp_takes_shortest_arc():
    # -rot_z(90) is the same rotation; the midpoint must still be rot_z(45), not a 135 deg path.
    mid = quat.slerp(IDENTITY, -rot_z(np.deg2rad(90.0)), 0.5)
    assert_same_rotation(mid, rot_z(np.deg2rad(45.0)))


def test_slerp_nearly_identical_inputs_is_stable():
    q1 = rot_z(1e-9)
    out = quat.slerp(IDENTITY, q1, 0.5)
    assert np.all(np.isfinite(out))
    assert_same_rotation(out, IDENTITY, atol=1e-8)


def test_slerp_matches_scipy(rng):
    from scipy.spatial.transform import Slerp

    q0, q1 = random_quats(rng, 2)
    ts = np.linspace(0.0, 1.0, 11)
    ref = Slerp([0.0, 1.0], quat.to_scipy(np.stack([q0, q1])))(ts)
    ours = quat.slerp(q0, q1, ts)
    assert_same_rotation(ours, quat.from_scipy(ref))


# --- Rotation vectors --------------------------------------------------------------------------


def test_rotvec_round_trip(rng):
    r = rng.normal(size=(100, 3))
    r = r / np.linalg.norm(r, axis=1, keepdims=True) * rng.uniform(0.0, np.pi - 1e-3, size=(100, 1))
    np.testing.assert_allclose(quat.to_rotvec(quat.from_rotvec(r)), r, atol=1e-9)


def test_rotvec_zero_and_tiny():
    np.testing.assert_allclose(quat.from_rotvec([0.0, 0.0, 0.0]), IDENTITY)
    np.testing.assert_allclose(quat.to_rotvec(IDENTITY), [0.0, 0.0, 0.0])
    tiny = np.array([1e-12, -2e-12, 3e-12])
    np.testing.assert_allclose(quat.to_rotvec(quat.from_rotvec(tiny)), tiny, atol=1e-20)


def test_rotvec_matches_scipy(rng):
    q = random_quats(rng, 50)
    np.testing.assert_allclose(quat.to_rotvec(q), quat.to_scipy(q).as_rotvec(), atol=1e-9)
    r = rng.normal(size=(50, 3))
    assert_same_rotation(quat.from_rotvec(r), quat.from_scipy(Rotation.from_rotvec(r)))


# --- Randomized checks against SciPy ------------------------------------------------------------


def test_multiply_matches_scipy(rng):
    q1 = random_quats(rng, 200)
    q2 = random_quats(rng, 200)
    ref = quat.from_scipy(quat.to_scipy(q1) * quat.to_scipy(q2))
    assert_same_rotation(quat.multiply(q1, q2), ref)


def test_rotate_vector_matches_scipy(rng):
    q = random_quats(rng, 200)
    v = rng.normal(size=(200, 3))
    np.testing.assert_allclose(quat.rotate_vector(q, v), quat.to_scipy(q).apply(v), atol=1e-12)


# --- Swing-twist decomposition -----------------------------------------------------------------

Y_AXIS = np.array([0.0, 1.0, 0.0])


def rot_x(angle_rad: float) -> np.ndarray:
    return np.array([np.cos(angle_rad / 2), np.sin(angle_rad / 2), 0.0, 0.0])


def rot_y(angle_rad: float) -> np.ndarray:
    return np.array([np.cos(angle_rad / 2), 0.0, np.sin(angle_rad / 2), 0.0])


def test_swing_twist_recomposes_random(rng):
    q = random_quats(rng, 200)
    axes = rng.normal(size=(200, 3))
    swing, twist = quat.swing_twist(q, axes)
    assert_same_rotation(quat.multiply(swing, twist), q)


def test_swing_twist_single_axis_broadcasts(rng):
    q = random_quats(rng, 50)
    swing, twist = quat.swing_twist(q, Y_AXIS)
    assert swing.shape == twist.shape == (50, 4)
    assert_same_rotation(quat.multiply(swing, twist), q)


def test_swing_axis_is_perpendicular_and_twist_axis_is_parallel(rng):
    q = random_quats(rng, 100)
    swing, twist = quat.swing_twist(q, Y_AXIS)
    np.testing.assert_allclose(swing[:, 2], 0.0, atol=1e-12)  # swing vector part has no Y component
    np.testing.assert_allclose(twist[:, [1, 3]], 0.0, atol=1e-12)  # twist vector part is along Y only


def test_pure_twist():
    q = rot_y(np.deg2rad(40.0))
    swing, twist = quat.swing_twist(q, Y_AXIS)
    assert_same_rotation(twist, q)
    assert_same_rotation(swing, IDENTITY)
    assert quat.twist_angle(q, Y_AXIS) == pytest.approx(np.deg2rad(40.0))


def test_pure_swing():
    q = rot_x(np.deg2rad(40.0))
    swing, twist = quat.swing_twist(q, Y_AXIS)
    assert_same_rotation(swing, q)
    assert_same_rotation(twist, IDENTITY)
    assert quat.twist_angle(q, Y_AXIS) == pytest.approx(0.0, abs=1e-12)


def test_twist_angle_is_signed():
    assert quat.twist_angle(rot_y(np.deg2rad(-30.0)), Y_AXIS) == pytest.approx(np.deg2rad(-30.0))
    # Same rotation written as -q gives the same angle (double cover).
    assert quat.twist_angle(-rot_y(np.deg2rad(-30.0)), Y_AXIS) == pytest.approx(np.deg2rad(-30.0))


def test_twist_angle_of_swing_then_twist():
    q = quat.multiply(rot_x(np.deg2rad(25.0)), rot_y(np.deg2rad(70.0)))
    assert quat.twist_angle(q, Y_AXIS) == pytest.approx(np.deg2rad(70.0))


def test_twist_angle_range_at_180():
    assert quat.twist_angle(rot_y(np.pi), Y_AXIS) == pytest.approx(np.pi)
    assert quat.twist_angle(-rot_y(np.pi), Y_AXIS) == pytest.approx(np.pi)


def test_swing_twist_singular_180_perpendicular():
    q = rot_x(np.pi)  # 180 deg about X: twist about Y undefined
    swing, twist = quat.swing_twist(q, Y_AXIS)
    assert np.all(np.isfinite(swing)) and np.all(np.isfinite(twist))
    assert_same_rotation(twist, IDENTITY)
    assert_same_rotation(swing, q)
    assert quat.twist_angle(q, Y_AXIS) == pytest.approx(0.0, abs=1e-12)


def test_swing_twist_rejects_zero_axis():
    with pytest.raises(ValueError):
        quat.swing_twist(IDENTITY, [0.0, 0.0, 0.0])


# --- Rotation matrices and two-vector rotation --------------------------------------------------


def test_matrix_round_trip_and_columns(rng):
    q = random_quats(rng, 50)
    R = quat.to_matrix(q)
    assert_same_rotation(quat.from_matrix(R), q)
    # Column j of R is child axis j expressed in the parent frame.
    np.testing.assert_allclose(R[:, :, 0], quat.rotate_vector(q, [1.0, 0.0, 0.0]), atol=1e-12)
    expected = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(quat.to_matrix(rot_z(np.pi / 2)), expected, atol=1e-12)


def test_matrix_nan_rows_propagate():
    R = np.stack([np.eye(3), np.full((3, 3), np.nan)])
    q = quat.from_matrix(R)
    np.testing.assert_allclose(q[0], IDENTITY)
    assert np.all(np.isnan(q[1]))
    assert np.all(np.isnan(quat.to_matrix(q)[1]))


def test_from_two_vectors(rng):
    a = rng.normal(size=(100, 3))
    b = rng.normal(size=(100, 3))
    q = quat.from_two_vectors(a, b)
    b_hat = b / np.linalg.norm(b, axis=1, keepdims=True)
    np.testing.assert_allclose(quat.rotate_vector(q, a / np.linalg.norm(a, axis=1, keepdims=True)), b_hat,
                               atol=1e-12)
    # Minimal rotation: no twist about a.
    np.testing.assert_allclose(quat.twist_angle(q, a), 0.0, atol=1e-9)


def test_from_two_vectors_parallel_and_antiparallel():
    np.testing.assert_allclose(quat.from_two_vectors([0, 0, 2], [0, 0, 1]), IDENTITY, atol=1e-12)
    for a in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]):
        q = quat.from_two_vectors(a, -np.array(a))
        np.testing.assert_allclose(quat.rotate_vector(q, a), -np.array(a), atol=1e-12)


# --- Shapes and edge cases ---------------------------------------------------------------------


def test_single_and_batched_shapes(rng):
    q = random_quats(rng, 5)
    assert quat.multiply(q[0], q[1]).shape == (4,)
    assert quat.multiply(q, q).shape == (5, 4)
    assert quat.multiply(q[0], q).shape == (5, 4)
    assert quat.rotate_vector(q, np.zeros(3)).shape == (5, 3)
    assert np.shape(quat.angle_between(q[0], q[1])) == ()


def test_normalize_rejects_zero():
    with pytest.raises(ValueError):
        quat.normalize([0.0, 0.0, 0.0, 0.0])


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        quat.conj([1.0, 0.0, 0.0])
