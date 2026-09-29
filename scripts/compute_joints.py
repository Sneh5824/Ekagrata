"""Compute camera-based joint angles, angular velocities and accelerations for a recorded session.

Writes <session>/motion/joints.parquet (source="cam") and <session>/motion/plots/*.png, and appends an entry
to session.json "derived". Raw session files are never modified.

Usage: uv run python scripts/compute_joints.py --session data/sessions/2026-09-29_live01_002
"""

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ekagrata.analysis.derivatives import resample_uniform, savgol_derivative, unwrap_nan
from ekagrata.analysis.filters import butter_filtfilt, interpolate_gaps, visibility_gate
from ekagrata.core.config import ConfigError, load_camera_config, load_joints_config
from ekagrata.io.session import code_version, read_session, write_json
from ekagrata.kinematics.angles import ANGLES, JOINT_QUATS, QUALITY, joint_angles_from_camera
from ekagrata.vision.landmark_map import N_LANDMARKS, mp_to_world

WRAPPING = {"shoulder_plane", "shoulder_rot", "forearm_pron", "racket_twist"}  # can jump by 2*pi
PLOT_ANGLES = ["shoulder_elev", "shoulder_plane", "shoulder_rot", "elbow_flex", "wrist_flex", "wrist_dev"]


def load_landmarks(lms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_sync_ns (N,), MediaPipe world landmarks (N, 33, 3), visibility (N, 33))."""
    world = np.stack([lms[[f"wlm{i}_{k}" for k in "xyz"]].to_numpy(float) for i in range(N_LANDMARKS)],
                     axis=1)
    vis = np.stack([lms[f"lm{i}_vis"].to_numpy(float) for i in range(N_LANDMARKS)], axis=1)
    return lms["t_sync_ns"].to_numpy(np.int64), world, vis


def compute(t_ns, world_mp, vis, mp_matrix, jc) -> tuple[pd.DataFrame, float]:
    """Full offline pipeline. Returns (joints table on a uniform grid, measured frame rate in Hz)."""
    fs_hz = 1e9 / float(np.median(np.diff(t_ns)))  # measured from recorded timestamps
    if not jc.butter_cutoff_hz < fs_hz / 2:
        raise ValueError(f"butter_cutoff_hz = {jc.butter_cutoff_hz} Hz must be below half the measured frame "
                         f"rate ({fs_hz:.2f} Hz / 2 = {fs_hz / 2:.2f} Hz); lower it in configs/joints.yaml")
    world = mp_to_world(visibility_gate(world_mp, vis, jc.visibility_threshold), mp_matrix)
    t_grid, x = resample_uniform(t_ns, world.reshape(len(t_ns), -1), fs_hz, max_gap_s=1.5 / fs_hz)
    x = interpolate_gaps(x, jc.max_gap_frames)
    x = butter_filtfilt(x, fs_hz, jc.butter_cutoff_hz, jc.butter_order)
    df = joint_angles_from_camera(x.reshape(len(t_grid), N_LANDMARKS, 3),
                                  np.deg2rad(jc.upper_arm_min_flex_deg),
                                  np.deg2rad(jc.shoulder_rot_singular_deg))
    cols = {"t_sync_ns": t_grid, "source": np.full(len(t_grid), "cam")}
    for name in ANGLES:
        a = df[name].to_numpy()
        cols[name] = a
        a_cont = unwrap_nan(a) if name in WRAPPING else a
        cols[f"{name}_vel"] = savgol_derivative(a_cont, fs_hz, jc.savgol_window, jc.savgol_polyorder, 1)
        cols[f"{name}_acc"] = savgol_derivative(a_cont, fs_hz, jc.savgol_window, jc.savgol_polyorder, 2)
    for q in JOINT_QUATS:
        for c in "wxyz":
            cols[f"{q}_{c}"] = df[f"{q}_{c}"].to_numpy()
    for k in ("x", "y", "z", "vx", "vy", "vz"):  # racket model arrives in M4/M8
        cols[f"racket_head_{k}"] = np.full(len(t_grid), np.nan)
    for q in QUALITY:
        cols[q] = df[q].to_numpy()
    return pd.DataFrame(cols), fs_hz


def plot(df: pd.DataFrame, fs_hz: float, session: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_s = (df["t_sync_ns"].to_numpy() - df["t_sync_ns"].iloc[0]) * 1e-9
    paths = []
    for suffix, unit, fname in (("", "deg", "joint_angles.png"), ("_vel", "deg/s", "joint_velocities.png")):
        fig, axes = plt.subplots(len(PLOT_ANGLES), 1, figsize=(11, 2.0 * len(PLOT_ANGLES)), sharex=True)
        for ax, name in zip(axes, PLOT_ANGLES, strict=True):
            y = np.rad2deg(df[name + suffix].to_numpy())
            ax.plot(t_s, y, lw=1)
            if not np.isfinite(y).any():
                ax.text(0.5, 0.5, "no valid samples (see compute_joints output)", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
            ax.set_ylabel(f"{name}{suffix}\n[{unit}]", fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("time since first frame [s]")
        fig.suptitle(f"REAL data, camera only - {session} - fs = {fs_hz:.2f} Hz (measured). "
                     "Camera cannot resolve the smash acceleration phase (SPEC §1.4).", fontsize=9)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", type=Path, required=True, help="session folder")
    p.add_argument("--config", type=Path, default=Path("configs/camera.yaml"), help="for mp_to_world")
    p.add_argument("--joints-config", type=Path, default=Path("configs/joints.yaml"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    if not (args.session / "session.json").is_file():
        print(f"ERROR: not a session folder (no session.json): {args.session}")
        return 2
    try:
        cam = load_camera_config(args.config)
        jc = load_joints_config(args.joints_config)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    meta, _, lms = read_session(args.session)
    if meta.get("handedness", "right") != "right":
        print("ERROR: left-handed sessions are not supported yet "
              "(mirroring per SPEC §3.1 is not implemented).")
        return 2
    if len(lms) < 2:
        print(f"ERROR: session has {len(lms)} landmark rows; need at least 2.")
        return 2

    t_ns, world_mp, vis = load_landmarks(lms)
    try:
        df, fs_hz = compute(t_ns, world_mp, vis, cam.mp_to_world, jc)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    motion = args.session / "motion"
    motion.mkdir(exist_ok=True)
    out = motion / "joints.parquet"
    df.to_parquet(out, index=False)  # derived file: may be regenerated
    plots = []
    if not args.no_plots:
        (motion / "plots").mkdir(exist_ok=True)
        plots = plot(df, fs_hz, meta.get("session_id", args.session.name), motion / "plots")

    meta_path = args.session / "session.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("derived", []).append({
        "file": "motion/joints.parquet",
        "script": "scripts/compute_joints.py",
        "code_version": code_version(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "fs_hz_measured": fs_hz,
        "mp_to_world": [list(r) for r in cam.mp_to_world],
        "params": dataclasses.asdict(jc),
    })
    write_json(meta_path, meta)

    has_pose = np.isfinite(world_mp[:, 0, 0])
    print(f"Session          : {args.session}")
    print(f"Input frames     : {len(t_ns)} ({int(has_pose.sum())} with a pose)")
    print(f"Frame rate       : {fs_hz:.2f} Hz (measured, median interval) -> "
          f"uniform grid of {len(df)} samples")
    print("Coverage (finite samples on the grid):")
    for name in ANGLES:
        cov = 100.0 * float(np.isfinite(df[name]).mean())
        note = " (not observable from camera)" if name in ("forearm_pron", "racket_twist") else ""
        note = " (quality: low)" if name.startswith("wrist") else note
        print(f"  {name:<15} {cov:5.1f} %{note}")
    if float(np.isfinite(df["shoulder_elev"]).mean()) < 0.5:
        hips = np.nanmedian(vis[:, [23, 24]])
        print(f"NOTE: shoulder angles need the torso frame, which needs both hips visible; median hip "
              f"visibility in this session = {hips:.2f} (threshold {jc.visibility_threshold}). "
              "Stand further back so the hips are in view.")
    print(f"Wrote            : {out}")
    for pth in plots:
        print(f"Plot             : {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
