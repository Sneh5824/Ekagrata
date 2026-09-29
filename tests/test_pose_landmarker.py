from pathlib import Path

import numpy as np
import pytest

from ekagrata.core.types import CameraFrame
from ekagrata.vision.pose_landmarker import (
    POSE_CONNECTIONS,
    create_landmarker,
    detect,
    has_pose,
    model_path,
    video_timestamp_ms,
)

MODELS = Path(__file__).resolve().parents[1] / "models"
LITE = model_path(MODELS, "lite")


def test_video_timestamp_ms_basic():
    assert video_timestamp_ms(5_000_000_000, 5_000_000_000, None) == 0
    assert video_timestamp_ms(5_033_400_000, 5_000_000_000, 0) == 33


def test_video_timestamp_ms_strictly_increasing_even_for_close_frames():
    t0 = 1_000_000_000
    stamps = [t0, t0 + 200_000, t0 + 400_000, t0 + 4_166_667, t0 + 4_166_667]  # sub-ms and repeated times
    out, last = [], None
    for t in stamps:
        last = video_timestamp_ms(t, t0, last)
        out.append(last)
    assert out == [0, 1, 2, 4, 5]
    assert np.all(np.diff(out) > 0)


def test_model_path_naming():
    assert model_path("models", "full").name == "pose_landmarker_full.task"


def test_pose_connections_indices_valid():
    assert len(POSE_CONNECTIONS) > 0
    assert all(0 <= a < 33 and 0 <= b < 33 for a, b in POSE_CONNECTIONS)


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        create_landmarker(tmp_path / "pose_landmarker_lite.task")


@pytest.mark.skipif(not LITE.is_file(), reason=f"model file absent: {LITE} (run scripts/download_models.py)")
def test_landmarker_on_blank_frames_detects_no_pose():
    landmarker = create_landmarker(LITE)
    try:
        for i in range(3):
            frame = CameraFrame(i, i * 33_000_000, None, 0, np.zeros((240, 320, 3), np.uint8))
            pose = detect(landmarker, frame, timestamp_ms=i * 33, t_sync_ns=frame.t_host_ns)
            assert pose.landmarks_img.shape == (33, 5)
            assert not has_pose(pose)
            assert np.all(np.isnan(pose.landmarks_img))
            assert pose.landmarks_world is None
            assert pose.frame_idx == i and pose.t_sync_ns == frame.t_host_ns
    finally:
        landmarker.close()
