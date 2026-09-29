"""Shared data types. Real, simulated and replayed sources must all produce these same types."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Source = Literal["real", "sim", "replay"]


@dataclass(frozen=True)
class ImuSample:
    """One IMU reading from a wearable node (BNO055 + ESP32)."""

    t_sync_ns: int  # canonical synchronized timeline
    t_node_us: int  # node ESP32 micros() at sampling
    t_rx_us: int  # USB receiver ESP32 micros() at reception
    t_host_ns: int  # host perf_counter_ns() at serial read
    node: str  # segment name: upper_arm, forearm, racket/hand, torso
    seq: int  # per-node packet sequence number
    q_wxyz: np.ndarray  # (4,) orientation q_world_sensor, [w, x, y, z]
    gyro_rad_s: np.ndarray  # (3,) angular velocity, rad/s
    acc_m_s2: np.ndarray  # (3,) acceleration including gravity, m/s^2
    lin_acc_m_s2: np.ndarray  # (3,) linear acceleration (gravity removed), m/s^2
    calib: tuple[int, int, int, int]  # BNO055 calibration status (sys, gyro, accel, mag), each 0..3
    flags: int  # bit field for node status
    source: Source


@dataclass(frozen=True)
class CameraFrame:
    """One camera image with its timestamps. Live and video-file sources both produce this type."""

    frame_idx: int
    t_host_ns: int  # live: perf_counter_ns() right after grab; file: equal to pts_ns
    pts_ns: int | None  # container presentation timestamp from file start (file mode), else None
    dropped_before: int  # frames dropped by the capture queue since the previous delivered frame
    image: np.ndarray  # (H, W, 3) uint8, BGR


@dataclass(frozen=True)
class PoseFrame:
    """MediaPipe PoseLandmarker output for one camera frame."""

    frame_idx: int
    t_host_ns: int
    t_sync_ns: int
    landmarks_img: np.ndarray  # (33, 5): x, y (normalized image coords), z, visibility, presence
    landmarks_world: np.ndarray | None  # (33, 3) in metres, or None if unavailable
