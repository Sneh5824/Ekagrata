"""UDP link to the Blender raw camera shadow: message format, world mapping and the never-raising sender."""

import json
import math
import socket

import numpy as np
import pytest

from ekagrata.transport.udp_twin import UdpTwinSender, decode, encode_landmarks, world_points
from ekagrata.vision.landmark_map import FACING_CAMERA, LM


class FakeClock:
    def __init__(self, step_ns):
        self.t = 0
        self.step_ns = step_ns

    def __call__(self):
        t = self.t
        self.t += self.step_ns
        return t


class FailingSocket:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def setblocking(self, flag):
        pass

    def sendto(self, data, addr):
        self.calls += 1
        raise self.exc

    def close(self):
        pass


def test_round_trip_within_rounding():
    points = {"r_wrist": [0.123456789, -1.0000049, 2.5], "nose": [0.0, 0.1, -0.2]}
    vis = {"r_wrist": 0.87654, "nose": 0.5}
    msg = decode(encode_landmarks(7, "real", 123_456_789_012, 123_456_999_999, points, vis))
    assert msg["v"] == 1 and msg["mode"] == "landmarks" and msg["source"] == "real"
    assert msg["seq"] == 7 and msg["t_sync_ns"] == 123_456_789_012 and msg["t_send_ns"] == 123_456_999_999
    for name, p in points.items():
        assert np.allclose(msg["points"][name], p, rtol=0, atol=0.5e-4 + 1e-12)  # rounded to 0.1 mm
        assert msg["vis"][name] == pytest.approx(vis[name], abs=0.5e-3 + 1e-12)


def test_non_finite_points_are_left_out_and_json_is_strict():
    data = encode_landmarks(1, "replay", 0, 1, {"a": [np.nan, 0, 0], "b": [1, 2, 3], "c": [math.inf, 0, 0]},
                            {"a": 0.9, "b": 0.8, "c": 0.7})

    def reject(token):
        raise AssertionError(f"non-standard JSON constant {token}")

    msg = json.loads(data.decode("utf-8"), parse_constant=reject)
    assert set(msg["points"]) == {"b"} and set(msg["vis"]) == {"b"}


def test_empty_frame_encodes_empty_dicts():
    msg = decode(encode_landmarks(1, "real", 0, 1, {}, {}))
    assert msg["points"] == {} and msg["vis"] == {}


@pytest.mark.parametrize("bad", [
    b"not json", b"[]", b'{"v":2,"mode":"landmarks"}',
    b'{"v":1,"seq":true,"mode":"landmarks","source":"real","t_sync_ns":0,"t_send_ns":0,"points":{},"vis":{}}',
    b'{"v":1,"seq":1,"mode":"landmarks","source":"sim","t_sync_ns":0,"t_send_ns":0,"points":{},"vis":{}}',
    b'{"v":1,"seq":1,"mode":"landmarks","source":"real","t_sync_ns":0,"t_send_ns":0,"points":{"a":[1,2]},'
    b'"vis":{}}',
])
def test_decode_rejects_invalid(bad):
    with pytest.raises(ValueError):
        decode(bad)


def test_encode_rejects_unknown_source():
    with pytest.raises(ValueError):
        encode_landmarks(1, "sim", 0, 1, {}, {})


def test_world_points_uses_the_one_mapping():
    world_mp = np.zeros((33, 3))
    world_mp[LM["r_wrist"]] = [1.0, 2.0, 3.0]  # MediaPipe: x right, y down, z away from camera
    vis = np.full(33, 0.9)
    pts, vs = world_points(world_mp, vis, ["r_wrist", "nose"], FACING_CAMERA)
    assert pts["r_wrist"] == pytest.approx([-3.0, 1.0, -2.0])  # X = -z, Y = +x, Z = -y
    assert vs == {"r_wrist": 0.9, "nose": 0.9}
    assert world_points(None, vis, ["r_wrist"], FACING_CAMERA) == ({}, {})


def test_rate_limit_with_fake_clock():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    clock = FakeClock(0)  # time only moves when the test sets it
    sender = UdpTwinSender("127.0.0.1", 9, 60.0, clock=clock, sock=sock)
    for i in range(100):  # calls at t = 0, 5, ..., 495 ms; 60 Hz -> at most one per 16.67 ms
        clock.t = i * 5_000_000
        sender.send_landmarks("real", 0, {}, {})
    sender.close()
    assert sender.sent + sender.send_errors == 25  # sends at 0, 20, 40, ..., 480 ms
    assert sender.rate_limited == 75
    if sender.send_errors == 0:
        assert sender.measured_rate_hz() == pytest.approx(50.0)


@pytest.mark.parametrize("exc", [OSError("network down"), BlockingIOError(), ConnectionResetError()])
def test_send_failure_never_raises(exc):
    fake = FailingSocket(exc)
    sender = UdpTwinSender("127.0.0.1", 9870, 1e6, clock=FakeClock(1_000_000), sock=fake)
    for _ in range(10):
        assert sender.send_landmarks("real", 0, {"a": [0, 0, 0]}, {"a": 1.0}) is False
    assert fake.calls == 10 and sender.send_errors == 10 and sender.sent == 0 and sender.seq == 0
    assert type(exc).__name__ in sender.last_error


def test_sender_survives_no_listener():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens on this port now
    sender = UdpTwinSender("127.0.0.1", port, 1e6)
    for i in range(200):
        sender.send_landmarks("real", i, {"a": [0.0, 0.0, float(i)]}, {"a": 1.0})
    sender.close()
    assert sender.sent + sender.send_errors + sender.rate_limited == 200


def test_listener_receives_consecutive_seq():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    sender = UdpTwinSender("127.0.0.1", rx.getsockname()[1], 1e6, clock=FakeClock(1_000_000))
    for i in range(5):
        assert sender.send_landmarks("replay", 1000 + i, {"b": [1, 2, 3]}, {"b": 0.9})
    msgs = [decode(rx.recv(65535)) for _ in range(5)]
    rx.close()
    sender.close()
    assert [m["seq"] for m in msgs] == [1, 2, 3, 4, 5]
    assert sender.last_sent_ns > sender.last_send_ns  # post-sendto stamp comes after t_send_ns
    assert [m["t_sync_ns"] for m in msgs] == [1000, 1001, 1002, 1003, 1004]
    assert all(b["t_send_ns"] > a["t_send_ns"] for a, b in zip(msgs, msgs[1:], strict=False))
