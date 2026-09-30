"""Framing checks (hips / hitting arm in view, wrist near or beyond the frame edges, rest gate)."""

import numpy as np
import pytest

from ekagrata.vision.framing import (
    GREEN,
    NEAR_TOP_Y,
    RED,
    YELLOW,
    draw_framing,
    framing_landmarks,
    framing_status,
    rest_gate,
    wrist_edge,
    wrist_edge_message,
)
from ekagrata.vision.landmark_map import LM


def pose(**overrides):
    """(33, 5) image landmarks, all at the image centre with visibility 0.9; overrides: name=(x, y, vis)."""
    lm = np.tile([0.5, 0.5, 0.0, 0.9, 0.9], (33, 1)).astype(np.float64)
    for name, (x, y, vis) in overrides.items():
        lm[LM[name], [0, 1, 3]] = (x, y, vis)
    return lm


NO_POSE = np.full((33, 5), np.nan)


def test_framing_landmarks_follow_side():
    assert framing_landmarks("right") == ("l_hip", "r_hip", "r_elbow", "r_wrist")
    assert framing_landmarks("left") == ("l_hip", "r_hip", "l_elbow", "l_wrist")
    with pytest.raises(ValueError):
        framing_landmarks("both")


def test_framing_status_visibility_and_in_frame():
    lm = pose(l_hip=(0.4, 1.05, 0.9),  # visible but below the bottom edge -> not ok
              r_hip=(0.6, 0.9, 0.49),  # in frame but visibility < 0.5 -> not ok
              r_elbow=(0.7, 0.4, 0.5))  # visibility exactly 0.5, in frame -> ok
    assert framing_status(lm) == {"l_hip": False, "r_hip": False, "r_elbow": True, "r_wrist": True}
    assert framing_status(NO_POSE) == {n: False for n in framing_landmarks("right")}


@pytest.mark.parametrize("y,expected", [
    (-0.01, "above_top"),
    (0.0, "near_top"),
    (0.05, "near_top"),
    (NEAR_TOP_Y - 1e-9, "near_top"),
    (NEAR_TOP_Y, None),  # exactly 0.10 is not "near" (y < 0.10)
    (0.5, None),
    (1.0, None),
    (1.01, "below_bottom"),
])
def test_wrist_edge_known_answers(y, expected):
    assert wrist_edge(pose(r_wrist=(0.5, y, 0.9))) == expected


def test_wrist_edge_uses_hitting_side_and_ignores_no_pose():
    lm = pose(l_wrist=(0.5, -0.2, 0.9))  # left wrist out, right wrist at the centre
    assert wrist_edge(lm, "right") is None and wrist_edge(lm, "left") == "above_top"
    assert wrist_edge(NO_POSE) is None


def test_wrist_edge_messages_and_colours():
    assert wrist_edge_message("above_top") == ("RIGHT WRIST ABOVE FRAME TOP", RED)
    assert wrist_edge_message("below_bottom") == ("RIGHT WRIST BELOW FRAME BOTTOM", RED)
    assert wrist_edge_message("near_top") == ("RIGHT WRIST NEAR FRAME TOP - racket cut off", YELLOW)
    assert wrist_edge_message(None) is None
    assert wrist_edge_message("above_top", "left")[0] == "LEFT WRIST ABOVE FRAME TOP"


def test_rest_gate_median_rule():
    samples = {
        "l_hip": [0.9, 0.1, 0.8],  # one bad frame, median 0.8 -> passes
        "r_hip": [0.02, 0.03, 0.9],  # median 0.03 -> fails
        "r_wrist": [0.6, np.nan, np.nan],  # NaN (no pose) counts as 0 -> median 0 -> fails
    }
    failed = dict(rest_gate(samples))
    assert set(failed) == {"r_hip", "r_wrist"}
    assert failed["r_hip"] == pytest.approx(0.03) and failed["r_wrist"] == 0.0
    assert rest_gate({"l_hip": [0.5, 0.5]}) == []  # median exactly 0.5 passes
    assert rest_gate({"l_hip": []}) == [("l_hip", 0.0)]


def test_draw_framing_marks_image():
    img = np.zeros((360, 640, 3), np.uint8)
    draw_framing(img, {"l_hip": True, "r_hip": False}, "near_top")
    assert (img == np.array(YELLOW, np.uint8)).all(axis=2).any()  # yellow banner
    assert (img == np.array(GREEN, np.uint8)).all(axis=2).any()  # l_hip OK label
    assert (img == np.array(RED, np.uint8)).all(axis=2).any()  # r_hip NOT IN VIEW label
    plain = np.zeros((360, 640, 3), np.uint8)
    draw_framing(plain, {}, None)
    assert not plain.any()  # nothing to draw
