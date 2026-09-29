"""YAML configuration loading with validation and clear error messages."""

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from ekagrata.vision.landmark_map import validate_rotation

BACKENDS = ("dshow", "msmf", "any")
MODEL_VARIANTS = ("lite", "full", "heavy")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or has invalid values."""


@dataclass(frozen=True)
class CameraConfig:
    device: int  # OpenCV camera index
    backend: str  # one of BACKENDS
    width: int  # requested frame width, px
    height: int  # requested frame height, px
    fps: int  # requested frame rate (the driver may ignore it; fps is always measured)
    fourcc: str  # requested pixel format, 4 characters, e.g. "MJPG"
    model_variant: str  # one of MODEL_VARIANTS
    queue_size: int  # capture queue length; oldest frame dropped when full
    mp_to_world: tuple  # 3x3 rotation, MediaPipe world axes -> EKAGRATA world (see vision/landmark_map.py)
    exposure_auto: bool | None = None  # None = leave the driver setting untouched
    exposure: float | None = None  # driver-specific units (DSHOW: log2 seconds); None = untouched


@dataclass(frozen=True)
class JointsConfig:
    """Parameters for scripts/compute_joints.py (M2). Angles in config are degrees (`_deg`); code uses rad."""

    visibility_threshold: float  # landmarks with visibility below this are set to NaN, in [0, 1]
    max_gap_frames: int  # NaN runs up to this many frames are linearly interpolated
    butter_cutoff_hz: float  # zero-phase Butterworth low-pass cutoff on landmark positions
    butter_order: int
    savgol_window: int  # odd number of samples for Savitzky-Golay derivatives
    savgol_polyorder: int
    upper_arm_min_flex_deg: float  # below this elbow flexion the upper-arm frame is undefined (SPEC §3.1)
    shoulder_rot_singular_deg: float  # shoulder_rot is NaN within this of elev 0 or 180 deg (SPEC §3.1)
    one_euro_min_cutoff_hz: float  # online One-Euro filter (live use)
    one_euro_beta: float
    one_euro_d_cutoff_hz: float


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path}: config file not found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
    return data


def _check_keys(path, data: dict, cls) -> None:
    all_keys = {f.name for f in fields(cls)}
    required = {f.name for f in fields(cls) if f.default is MISSING}
    unknown = sorted(set(data) - all_keys)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(all_keys)}")
    missing = sorted(k for k in required if k not in data)
    if missing:
        raise ConfigError(f"{path}: missing required key(s) {missing}")


def _require(path, key, value, typ, cond=None, msg=""):
    # bool is a subclass of int; never accept it where a number is expected.
    if isinstance(value, bool) and typ is not bool or not isinstance(value, typ):
        raise ConfigError(f"{path}: '{key}' must be {getattr(typ, '__name__', typ)}, got {value!r}")
    if cond is not None and not cond(value):
        raise ConfigError(f"{path}: '{key}' {msg}, got {value!r}")


def load_camera_config(path) -> CameraConfig:
    data = load_yaml(path)
    _check_keys(path, data, CameraConfig)
    _require(path, "device", data["device"], int, lambda v: v >= 0, "must be >= 0")
    _require(path, "backend", data["backend"], str, lambda v: v in BACKENDS, f"must be one of {BACKENDS}")
    for key in ("width", "height", "fps", "queue_size"):
        _require(path, key, data[key], int, lambda v: v > 0, "must be > 0")
    _require(path, "fourcc", data["fourcc"], str, lambda v: len(v) == 4, "must be exactly 4 characters")
    _require(
        path, "model_variant", data["model_variant"], str, lambda v: v in MODEL_VARIANTS,
        f"must be one of {MODEL_VARIANTS}",
    )
    if data.get("exposure_auto") is not None:
        _require(path, "exposure_auto", data["exposure_auto"], bool)
    if data.get("exposure") is not None:
        _require(path, "exposure", data["exposure"], (int, float))
        data["exposure"] = float(data["exposure"])
    try:
        m = validate_rotation(data["mp_to_world"])
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path}: 'mp_to_world' invalid: {exc}") from exc
    data["mp_to_world"] = tuple(tuple(float(v) for v in row) for row in m)
    return CameraConfig(**data)


def load_joints_config(path) -> JointsConfig:
    data = load_yaml(path)
    _check_keys(path, data, JointsConfig)
    number = (int, float)
    _require(path, "visibility_threshold", data["visibility_threshold"], number, lambda v: 0 <= v <= 1,
             "must be in [0, 1]")
    _require(path, "max_gap_frames", data["max_gap_frames"], int, lambda v: v >= 0, "must be >= 0")
    _require(path, "butter_cutoff_hz", data["butter_cutoff_hz"], number, lambda v: v > 0, "must be > 0")
    _require(path, "butter_order", data["butter_order"], int, lambda v: 1 <= v <= 8, "must be in 1..8")
    _require(path, "savgol_window", data["savgol_window"], int, lambda v: v >= 3 and v % 2 == 1,
             "must be an odd integer >= 3")
    _require(path, "savgol_polyorder", data["savgol_polyorder"], int,
             lambda v: 2 <= v < data["savgol_window"], "must be >= 2 (2nd derivative) and < savgol_window")
    for key in ("upper_arm_min_flex_deg", "shoulder_rot_singular_deg"):
        _require(path, key, data[key], number, lambda v: 0 <= v < 90, "must be in [0, 90)")
    for key in ("one_euro_min_cutoff_hz", "one_euro_d_cutoff_hz"):
        _require(path, key, data[key], number, lambda v: v > 0, "must be > 0")
    _require(path, "one_euro_beta", data["one_euro_beta"], number, lambda v: v >= 0, "must be >= 0")
    return JointsConfig(**{k: (float(v) if isinstance(v, float) else v) for k, v in data.items()})
