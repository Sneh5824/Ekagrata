import numpy as np
import pytest

from ekagrata.vision.landmark_map import FACING_CAMERA, mp_to_world, validate_rotation


def test_facing_camera_known_answers():
    # MediaPipe: x image-right, y down, z away from camera.
    np.testing.assert_allclose(mp_to_world([0, -1, 0], FACING_CAMERA), [0, 0, 1])  # image up -> world up
    np.testing.assert_allclose(mp_to_world([0, 0, -1], FACING_CAMERA), [1, 0, 0])  # toward camera -> +X
    np.testing.assert_allclose(mp_to_world([1, 0, 0], FACING_CAMERA), [0, 1, 0])  # image right -> +Y


def test_facing_camera_is_proper_rotation():
    assert np.linalg.det(FACING_CAMERA) == pytest.approx(1.0)
    validate_rotation(FACING_CAMERA)


def test_batched_and_nan():
    p = np.array([[[0.0, -1, 0], [np.nan, 0, 0]]])
    out = mp_to_world(p, FACING_CAMERA)
    assert out.shape == (1, 2, 3)
    np.testing.assert_allclose(out[0, 0], [0, 0, 1])
    assert np.all(np.isnan(out[0, 1, [1]]))


def test_mirror_rejected():
    mirror = FACING_CAMERA.copy()
    mirror[1] *= -1
    with pytest.raises(ValueError, match="det"):
        validate_rotation(mirror)


def test_non_orthonormal_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation(np.eye(3) * 2)
    with pytest.raises(ValueError, match="3x3"):
        validate_rotation(np.eye(2))
