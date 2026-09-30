"""Camera tilt diagnostic (torso direction -> pitch/roll estimate; not applied to data)."""

import numpy as np
import pytest

from ekagrata.vision.camera_tilt import estimate_camera_tilt, torso_vector
from ekagrata.vision.landmark_map import LM


@pytest.mark.parametrize("pitch", [-20.0, 0.0, 7.5, 30.0])
def test_pitch_known_answer(pitch):
    p = np.radians(pitch)
    t = 0.5 * np.array([np.sin(p), 0.0, np.cos(p)])  # true up seen by a camera pitched down by `pitch`
    est_pitch, est_roll, n = estimate_camera_tilt([t] * 5)
    assert est_pitch == pytest.approx(pitch) and est_roll == pytest.approx(0.0, abs=1e-12) and n == 5


@pytest.mark.parametrize("roll", [-10.0, 4.0])
def test_roll_known_answer(roll):
    r = np.radians(roll)
    t = np.array([0.0, -np.sin(r), np.cos(r)])  # clockwise roll: up appears tilted toward image left (-Y)
    est_pitch, est_roll, _ = estimate_camera_tilt([t])
    assert est_roll == pytest.approx(roll) and est_pitch == pytest.approx(0.0, abs=1e-12)


def test_median_ignores_outliers_and_nan():
    good = [0.0, 0.0, 0.5]
    est_pitch, est_roll, n = estimate_camera_tilt([good, good, [0.5, 0.0, 0.0], [np.nan] * 3, good])
    assert (est_pitch, est_roll, n) == (0.0, 0.0, 4)
    assert np.isnan(estimate_camera_tilt([[np.nan] * 3])[0]) and estimate_camera_tilt([])[2] == 0


def test_torso_vector():
    w = np.zeros((33, 3))
    w[LM["l_hip"]], w[LM["r_hip"]] = [0, 0.1, 0], [0, -0.1, 0]
    w[LM["l_shoulder"]], w[LM["r_shoulder"]] = [0.02, 0.2, 0.5], [0.0, -0.2, 0.5]
    assert np.allclose(torso_vector(w), [0.01, 0.0, 0.5])
