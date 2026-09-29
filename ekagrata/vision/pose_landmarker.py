"""MediaPipe Tasks API PoseLandmarker in VIDEO running mode (never the legacy mp.solutions.pose)."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ekagrata.core.types import CameraFrame, PoseFrame

N_LANDMARKS = 33
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


def model_path(models_dir, variant: str) -> Path:
    return Path(models_dir) / f"pose_landmarker_{variant}.task"


def create_landmarker(
    path,
    num_poses: int = 1,
    min_detection: float = 0.5,
    min_presence: float = 0.5,
    min_tracking: float = 0.5,
) -> vision.PoseLandmarker:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"model file not found: {path}. Run: uv run python scripts/download_models.py"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection,
        min_pose_presence_confidence=min_presence,
        min_tracking_confidence=min_tracking,
    )
    return vision.PoseLandmarker.create_from_options(options)


def video_timestamp_ms(t_ns: int, t0_ns: int, last_ms: int | None) -> int:
    """VIDEO-mode timestamp in ms from the first frame. MediaPipe requires strictly increasing values, so a
    frame that would repeat or go back in time is bumped to last_ms + 1."""
    ms = (int(t_ns) - int(t0_ns)) // 1_000_000
    if last_ms is not None and ms <= last_ms:
        ms = last_ms + 1
    return ms


def _value(v) -> float:
    return np.nan if v is None else float(v)


def detect(landmarker, frame: CameraFrame, timestamp_ms: int, t_sync_ns: int) -> PoseFrame:
    """Run the landmarker on one BGR frame. No pose -> image landmarks all NaN and world landmarks None."""
    rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, timestamp_ms)
    img = np.full((N_LANDMARKS, 5), np.nan)
    world = None
    if result.pose_landmarks:
        img = np.array([
            [_value(lm.x), _value(lm.y), _value(lm.z), _value(lm.visibility), _value(lm.presence)]
            for lm in result.pose_landmarks[0]
        ])
        if result.pose_world_landmarks:
            world = np.array([
                [_value(lm.x), _value(lm.y), _value(lm.z)] for lm in result.pose_world_landmarks[0]
            ])
    return PoseFrame(frame.frame_idx, frame.t_host_ns, t_sync_ns, img, world)


def has_pose(pose: PoseFrame) -> bool:
    return not np.all(np.isnan(pose.landmarks_img[:, 0]))


def draw_skeleton(image: np.ndarray, pose: PoseFrame, vis_threshold: float = 0.5) -> None:
    """Draw landmarks and connections in place on a BGR image (preview only)."""
    if not has_pose(pose):
        return
    h, w = image.shape[:2]
    pts = pose.landmarks_img
    ok = np.nan_to_num(pts[:, 3], nan=1.0) >= vis_threshold
    xy = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for a, b in POSE_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(image, xy[a], xy[b], (0, 255, 0), 2)
    for i, p in enumerate(xy):
        if ok[i]:
            cv2.circle(image, p, 3, (0, 0, 255), -1)
