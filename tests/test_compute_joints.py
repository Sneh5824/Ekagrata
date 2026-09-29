"""End-to-end test of the scripts/compute_joints.py pipeline on a synthetic (SIMULATED) recording."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ekagrata.core import quat
from ekagrata.core.config import load_joints_config
from ekagrata.vision.landmark_map import FACING_CAMERA
from tests.test_angles import arm_pose, skeleton_from_frames

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compute_joints", REPO / "scripts" / "compute_joints.py")
compute_joints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compute_joints)


def synthetic_session(fs=15.0, seconds=6.0, f_hz=0.5, jitter_ms=3.0, seed=0):
    """Right arm elevating sinusoidally (plane 0.6 rad); MediaPipe-axis landmarks; jittered timestamps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t_s = np.arange(n) / fs + rng.uniform(-jitter_ms, jitter_ms, n) * 1e-3
    t_s[0] = 0.0
    elev = 1.2 + 0.6 * np.sin(2 * np.pi * f_hz * t_s)
    q_t = quat.from_matrix(np.stack([np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1]], axis=-1))
    world = np.stack([skeleton_from_frames(q_t, *arm_pose(q_t, 0.6, e, 0.3, 1.0)) for e in elev])
    world_mp = world @ FACING_CAMERA  # inverse of mp_to_world (M orthonormal)
    vis = np.ones(world.shape[:2])
    return (t_s * 1e9).astype(np.int64), world_mp, vis, elev, t_s


def test_pipeline_tracks_known_elevation_and_velocity():
    jc = load_joints_config(REPO / "configs" / "joints.yaml")
    t_ns, world_mp, vis, elev, t_s = synthetic_session()
    df, fs = compute_joints.compute(t_ns, world_mp, vis, FACING_CAMERA, jc)
    assert fs == pytest.approx(15.0, rel=0.02)
    assert set(df["source"]) == {"cam"}
    tg = (df["t_sync_ns"].to_numpy() - t_ns[0]) * 1e-9
    core = slice(10, -10)  # filter / derivative edge effects excluded
    true_elev = 1.2 + 0.6 * np.sin(np.pi * tg)
    true_vel = 0.6 * np.pi * np.cos(np.pi * tg)
    # Tolerances: timestamp jitter (+-3 ms) and 5 Hz low-pass at 15 Hz sampling (SIMULATED check).
    np.testing.assert_allclose(df["shoulder_elev"].to_numpy()[core], true_elev[core], atol=0.02)
    np.testing.assert_allclose(df["shoulder_elev_vel"].to_numpy()[core], true_vel[core], atol=0.1)
    np.testing.assert_allclose(df["shoulder_plane"].to_numpy()[core], 0.6, atol=0.01)
    np.testing.assert_allclose(df["elbow_flex"].to_numpy()[core], 1.0, atol=0.01)
    assert df["forearm_pron"].isna().all() and df["racket_twist"].isna().all()
    assert np.all(np.diff(df["t_sync_ns"].to_numpy()) > 0)


def test_pipeline_invisible_hips_give_nan_shoulder_not_crash():
    jc = load_joints_config(REPO / "configs" / "joints.yaml")
    t_ns, world_mp, vis, _, _ = synthetic_session()
    vis[:, [23, 24]] = 0.0
    df, _ = compute_joints.compute(t_ns, world_mp, vis, FACING_CAMERA, jc)
    assert df["shoulder_elev"].isna().all()
    assert np.isfinite(df["elbow_flex"]).mean() > 0.9


def test_pipeline_rejects_cutoff_above_nyquist():
    import dataclasses

    jc = dataclasses.replace(load_joints_config(REPO / "configs" / "joints.yaml"), butter_cutoff_hz=8.0)
    t_ns, world_mp, vis, _, _ = synthetic_session()
    with pytest.raises(ValueError, match="below half the measured frame rate"):
        compute_joints.compute(t_ns, world_mp, vis, FACING_CAMERA, jc)
