import cv2
import numpy as np
import pytest

from ekagrata.sources.pose import VideoFileSource, pts_report


def make_video(path, n_frames=30, fps=30.0, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 8) % 256, dtype=np.uint8))
    writer.release()
    return path


def test_video_file_source_reads_all_frames_with_pts(tmp_path):
    src = VideoFileSource(make_video(tmp_path / "clip.mp4", n_frames=30, fps=30.0))
    info = src.info()
    frames = list(src.frames())
    src.close()
    assert info["reported"]["w"] == 64 and info["reported"]["h"] == 48
    assert [f.frame_idx for f in frames] == list(range(30))
    pts = np.array([f.pts_ns for f in frames])
    assert pts[0] == 0
    assert np.all(np.diff(pts) > 0)  # strictly increasing
    np.testing.assert_allclose(np.diff(pts) * 1e-6, 1000.0 / 30.0, atol=0.5)  # ms
    assert all(f.t_host_ns == f.pts_ns and f.dropped_before == 0 for f in frames)
    assert frames[0].image.shape == (48, 64, 3)


def test_pts_report_on_file(tmp_path):
    src = VideoFileSource(make_video(tmp_path / "clip.mp4", n_frames=25, fps=25.0))
    rep = pts_report([f.pts_ns for f in src.frames()])
    src.close()
    assert rep["n_frames"] == 25
    assert rep["measured_fps"] == pytest.approx(25.0, rel=0.01)
    assert rep["non_increasing"] == 0 and rep["irregular"] == 0


def test_pts_report_known_answer():
    t = np.array([0, 10, 20, 30, 45, 55], dtype=np.int64) * 1_000_000  # ms -> ns
    rep = pts_report(t)
    assert rep["n_frames"] == 6
    assert rep["interval_ms_median"] == pytest.approx(10.0)
    assert rep["interval_ms_max"] == pytest.approx(15.0)
    assert rep["irregular"] == 1  # 15 ms deviates by 50% from the 10 ms median
    assert rep["measured_fps"] == pytest.approx(5 / 0.055)


def test_pts_report_flags_non_increasing():
    rep = pts_report(np.array([0, 10, 10, 20]) * 1_000_000)
    assert rep["non_increasing"] == 1


def test_pts_report_short_input():
    assert pts_report([]) == {"n_frames": 0}
    assert pts_report([5]) == {"n_frames": 1}


def test_missing_video_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="video file not found"):
        VideoFileSource(tmp_path / "missing.mp4")
