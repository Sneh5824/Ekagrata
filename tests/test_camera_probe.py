"""Selection logic of scripts/camera_probe.py (the measurement itself needs real hardware)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("camera_probe", REPO / "scripts" / "camera_probe.py")
camera_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(camera_probe)


def row(backend="dshow", w=1280, h=720, exposure=-6.0, fps=29.9, med=32.0, bright=60.0, dups=0,
        rep_exp=None, shape=None):
    return {"status": "ok", "backend": backend, "w": w, "h": h, "codec": "MJPG", "exposure": exposure,
            "fps_meas": fps, "dt_med_ms": med, "brightness": bright, "dups": dups,
            "rep_exp": exposure if rep_exp is None and exposure != "auto" else (rep_exp or -6.0),
            "shape": shape or f"{w}x{h}"}


def test_rejects_duplicates_bursts_dark_and_wrong_size():
    rows = [
        row(exposure="auto", fps=29.8, med=56.0, dups=45),  # MSMF-style padding with duplicates
        row(exposure=-7.0, bright=25.0),  # too dark
        row(shape="640x480"),  # driver ignored the size
        row(exposure="auto", fps=14.96, med=64.1, bright=140.0),  # usable but slow
    ]
    assert camera_probe.choose_best(rows, min_brightness=40.0)["fps_meas"] == 14.96


def test_prefers_resolution_then_readback_then_shorter_exposure():
    rows = [
        row(w=640, h=480, exposure=-6.0, fps=30.1),
        row(backend="msmf", exposure=-5.0, fps=30.08, rep_exp=-6.0),  # reports wrong exposure
        row(backend="dshow", exposure=-5.0, fps=29.9, bright=96.0),
        row(backend="dshow", exposure=-6.0, fps=29.92, bright=58.0),
    ]
    best = camera_probe.choose_best(rows, min_brightness=40.0)
    assert (best["backend"], best["w"], best["exposure"]) == ("dshow", 1280, -6.0)


def test_nothing_usable():
    assert camera_probe.choose_best([row(bright=10.0)], min_brightness=40.0) is None
    assert camera_probe.choose_best([{"status": "open failed"}], min_brightness=40.0) is None
