from pathlib import Path

import pytest

from ekagrata.core.config import (
    CameraConfig,
    ConfigError,
    JointsConfig,
    load_camera_config,
    load_joints_config,
)

REPO = Path(__file__).resolve().parents[1]

VALID = """
device: 0
backend: dshow
width: 1280
height: 720
fps: 60
fourcc: MJPG
model_variant: full
queue_size: 4
mp_to_world: [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
"""


def write(tmp_path, text):
    p = tmp_path / "camera.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_repo_camera_yaml_loads():
    cfg = load_camera_config(REPO / "configs" / "camera.yaml")
    assert isinstance(cfg, CameraConfig)


def test_valid_minimal(tmp_path):
    cfg = load_camera_config(write(tmp_path, VALID))
    assert cfg.width == 1280 and cfg.fourcc == "MJPG"
    assert cfg.exposure_auto is None and cfg.exposure is None


def test_optional_exposure(tmp_path):
    cfg = load_camera_config(write(tmp_path, VALID + "exposure_auto: false\nexposure: -6\n"))
    assert cfg.exposure_auto is False and cfg.exposure == -6.0


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_camera_config(tmp_path / "nope.yaml")


def test_missing_key_is_named(tmp_path):
    text = VALID.replace("queue_size: 4\n", "")
    with pytest.raises(ConfigError, match="queue_size"):
        load_camera_config(write(tmp_path, text))


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key.*led_roi"):
        load_camera_config(write(tmp_path, VALID + "led_roi: 1\n"))


def test_mp_to_world_parsed_and_mirror_rejected(tmp_path):
    cfg = load_camera_config(write(tmp_path, VALID))
    assert cfg.mp_to_world == ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    mirror = VALID.replace("[1, 0, 0]", "[-1, 0, 0]")
    with pytest.raises(ConfigError, match="mp_to_world.*det"):
        load_camera_config(write(tmp_path, mirror))
    with pytest.raises(ConfigError, match="mp_to_world"):
        load_camera_config(write(tmp_path, VALID.replace("[0, -1, 0]]", "[0, -1]]")))


def test_repo_joints_yaml_loads():
    cfg = load_joints_config(REPO / "configs" / "joints.yaml")
    assert isinstance(cfg, JointsConfig)
    assert cfg.savgol_window % 2 == 1


@pytest.mark.parametrize(
    "key,bad",
    [("visibility_threshold", 1.5), ("savgol_window", 6), ("savgol_polyorder", 9), ("butter_order", 0),
     ("upper_arm_min_flex_deg", 95.0)],
)
def test_joints_bad_values(tmp_path, key, bad):
    import yaml

    data = yaml.safe_load((REPO / "configs" / "joints.yaml").read_text(encoding="utf-8"))
    data[key] = bad
    p = tmp_path / "joints.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_joints_config(p)


@pytest.mark.parametrize(
    "old,new,key",
    [
        ("width: 1280", "width: -1", "width"),
        ("width: 1280", "width: 12.5", "width"),
        ("fps: 60", "fps: true", "fps"),
        ("backend: dshow", "backend: v4l2", "backend"),
        ("fourcc: MJPG", "fourcc: MJ", "fourcc"),
        ("model_variant: full", "model_variant: huge", "model_variant"),
    ],
)
def test_bad_values_name_the_key(tmp_path, old, new, key):
    with pytest.raises(ConfigError, match=f"'{key}'"):
        load_camera_config(write(tmp_path, VALID.replace(old, new)))


def test_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_camera_config(write(tmp_path, "device: [0,\n"))


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_camera_config(write(tmp_path, "- 1\n- 2\n"))
