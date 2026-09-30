"""YAML configuration loading with validation and clear error messages."""

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from ekagrata.vision.landmark_map import LM, validate_rotation

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
    # Camera tilt w.r.t. gravity, measured with a spirit-level app. Recorded only: NOT yet applied anywhere
    # (EKAGRATA "Z up" is currently camera-up; gravity alignment is an open item, SPEC M2/M7).
    camera_pitch_deg: float = 0.0  # positive = camera looks down
    camera_roll_deg: float = 0.0  # positive = camera rotated clockwise, seen from behind the camera


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
    for key in ("camera_pitch_deg", "camera_roll_deg"):
        if key in data:
            _require(path, key, data[key], (int, float), lambda v: -90 < v < 90,
                     "must be in (-90, 90) degrees")
            data[key] = float(data[key])
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


HITTING_SIDES = ("right", "left")


@dataclass(frozen=True)
class MatchedCamera:
    """Blender camera matching the real camera (M5-preview side-by-side view). MediaPipe world axes follow the
    camera, so the real camera looks exactly along -X (EKAGRATA world) from the hips: yaw 0 = matched view,
    any other yaw = alternative viewpoint."""

    distance_m: float  # horizontal distance hips -> camera lens (tape measure)
    height_m: float  # camera lens height minus hip height (tape measure; negative if the camera is lower)
    yaw_deg: float  # 0 = matched; otherwise an alternative viewpoint rotated about +Z around the hips
    horizontal_fov_deg: float  # real camera's horizontal field of view (measure: docs/blender.md)


@dataclass(frozen=True)
class BlenderConfig:
    """Blender raw camera shadow (M5-preview): Blender binary, UDP link and figure placement."""

    blender_exe: str  # path to blender.exe (pinned version below)
    blender_version: str  # expected bpy.app.version_string, e.g. "5.0.1"
    udp_host: str  # loopback address the sender targets
    udp_port: int
    send_rate_max_hz: float  # sender rate limit
    hitting_side: str  # one of HITTING_SIDES
    landmarks: tuple  # landmark names (keys of ekagrata.vision.landmark_map.LM)
    placement_offset_m: tuple  # (3,) display offset of the figure on the court, metres
    matched_camera: MatchedCamera  # side-by-side view camera placement


def load_blender_config(path) -> BlenderConfig:
    data = load_yaml(path)
    _check_keys(path, data, BlenderConfig)
    for key in ("blender_exe", "blender_version"):
        _require(path, key, data[key], str, lambda v: v.strip() != "", "must not be empty")
    _require(path, "udp_host", data["udp_host"], str, lambda v: v.startswith("127."),
             "must be a loopback address (127.x.x.x)")
    _require(path, "udp_port", data["udp_port"], int, lambda v: 1024 <= v <= 65535, "must be in 1024..65535")
    _require(path, "send_rate_max_hz", data["send_rate_max_hz"], (int, float), lambda v: v > 0, "must be > 0")
    _require(path, "hitting_side", data["hitting_side"], str, lambda v: v in HITTING_SIDES,
             f"must be one of {HITTING_SIDES}")
    _require(path, "landmarks", data["landmarks"], list,
             lambda v: len(v) > 0 and len(set(v)) == len(v) and all(n in LM for n in v),
             f"must be a non-empty list of unique names from {sorted(LM)}")
    _require(path, "placement_offset_m", data["placement_offset_m"], list,
             lambda v: len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                           for x in v),
             "must be a list of 3 numbers")
    data["send_rate_max_hz"] = float(data["send_rate_max_hz"])
    data["landmarks"] = tuple(data["landmarks"])
    data["placement_offset_m"] = tuple(float(x) for x in data["placement_offset_m"])
    data["matched_camera"] = _load_matched_camera(path, data["matched_camera"])
    return BlenderConfig(**data)


def _load_matched_camera(path, mc) -> MatchedCamera:
    if not isinstance(mc, dict):
        raise ConfigError(f"{path}: 'matched_camera' must be a mapping, got {mc!r}")
    _check_keys(f"{path} [matched_camera]", mc, MatchedCamera)
    number = (int, float)
    _require(path, "matched_camera.distance_m", mc["distance_m"], number, lambda v: v > 0, "must be > 0")
    _require(path, "matched_camera.height_m", mc["height_m"], number, lambda v: -5 < v < 5,
             "must be in (-5, 5) m")
    _require(path, "matched_camera.yaw_deg", mc["yaw_deg"], number, lambda v: -180 < v <= 180,
             "must be in (-180, 180]")
    _require(path, "matched_camera.horizontal_fov_deg", mc["horizontal_fov_deg"], number,
             lambda v: 1 < v < 179, "must be in (1, 179) degrees")
    return MatchedCamera(**{k: float(v) for k, v in mc.items()})
