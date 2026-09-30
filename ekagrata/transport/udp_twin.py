"""UDP link to Blender for the raw camera shadow (M5-preview; NOT the Digital Twin).

Message (JSON, UTF-8, one per datagram):
    {"v": 1, "seq": int, "mode": "landmarks", "source": "real"|"replay", "t_sync_ns": int, "t_send_ns": int,
     "points": {name: [x, y, z]}, "vis": {name: float}}
`points` are EKAGRATA world (Z up, metres) relative to the MediaPipe hip-midpoint origin. Landmarks that are
missing or non-finite are left out of both `points` and `vis`; a frame without a pose sends empty dicts.
`source` tells the receiver whether `t_sync_ns` is a live host-clock capture time ("real") or a recorded one
("replay"). `t_send_ns` is `time.perf_counter_ns()` of the sender right before `sendto`.
"""

import json
import math
import socket
import time

import numpy as np

from ekagrata.vision.landmark_map import LM, mp_to_world

MSG_VERSION = 1
MODE_LANDMARKS = "landmarks"
SOURCES = ("real", "replay")
COORD_DECIMALS = 4  # 0.1 mm
VIS_DECIMALS = 3
MAX_DATAGRAM = 65507


def encode_landmarks(seq: int, source: str, t_sync_ns: int, t_send_ns: int, points: dict, vis: dict) -> bytes:
    """Serialise one landmarks message. Non-finite points (and their vis) are left out."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    pts, vs = {}, {}
    for name, p in points.items():
        xyz = [float(c) for c in p]
        if len(xyz) != 3 or not all(math.isfinite(c) for c in xyz):
            continue
        pts[name] = [round(c, COORD_DECIMALS) for c in xyz]
        v = float(vis.get(name, math.nan))
        if math.isfinite(v):
            vs[name] = round(v, VIS_DECIMALS)
    msg = {"v": MSG_VERSION, "seq": int(seq), "mode": MODE_LANDMARKS, "source": source,
           "t_sync_ns": int(t_sync_ns), "t_send_ns": int(t_send_ns), "points": pts, "vis": vs}
    data = json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_DATAGRAM:
        raise ValueError(f"message is {len(data)} bytes, larger than one UDP datagram ({MAX_DATAGRAM})")
    return data


def decode(data: bytes) -> dict:
    """Parse and validate one message; ValueError if it is not a valid v1 landmarks message."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not UTF-8 JSON: {exc}") from exc
    if not isinstance(msg, dict) or msg.get("v") != MSG_VERSION or msg.get("mode") != MODE_LANDMARKS:
        raise ValueError("not a v1 landmarks message")
    for key in ("seq", "t_sync_ns", "t_send_ns"):
        if not isinstance(msg.get(key), int) or isinstance(msg.get(key), bool):
            raise ValueError(f"'{key}' must be an integer")
    if msg.get("source") not in SOURCES:
        raise ValueError(f"'source' must be one of {SOURCES}")
    if not isinstance(msg.get("points"), dict) or not isinstance(msg.get("vis"), dict):
        raise ValueError("'points' and 'vis' must be objects")
    for name, p in msg["points"].items():
        if not (isinstance(p, list) and len(p) == 3 and all(isinstance(c, (int, float)) for c in p)):
            raise ValueError(f"point {name!r} must be [x, y, z]")
    return msg


def world_points(world_mp, vis, names, mp_matrix) -> tuple[dict, dict]:
    """Select `names` from one frame of MediaPipe world landmarks (33, 3) and visibilities (33,), and map them
    to EKAGRATA world with `mp_matrix` (the ONE conversion in vision/landmark_map.py). `world_mp is None`
    (no pose) gives empty dicts."""
    if world_mp is None:
        return {}, {}
    idx = [LM[n] for n in names]
    pts = mp_to_world(np.asarray(world_mp, dtype=np.float64)[idx], mp_matrix)
    v = np.asarray(vis, dtype=np.float64)[idx]
    return ({n: pts[i].tolist() for i, n in enumerate(names)}, {n: float(v[i]) for i, n in enumerate(names)})


class UdpTwinSender:
    """Non-blocking, rate-limited UDP sender. `send_landmarks` never raises on a send failure: failures are
    counted in `send_errors`. `seq` increases by one per datagram actually handed to the OS, so a gap seen by
    the receiver is a loss in transport."""

    def __init__(self, host: str, port: int, max_rate_hz: float, clock=time.perf_counter_ns, sock=None):
        if max_rate_hz <= 0:
            raise ValueError("max_rate_hz must be > 0")
        self.addr = (host, int(port))
        self.min_interval_ns = int(round(1e9 / max_rate_hz))
        self.clock = clock
        self.sock = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.seq = 0
        self.sent = 0
        self.rate_limited = 0
        self.send_errors = 0
        self.last_error = ""
        self.first_send_ns: int | None = None
        self.last_send_ns: int | None = None

    def send_landmarks(self, source: str, t_sync_ns: int, points: dict, vis: dict) -> bool:
        """Send one message unless rate-limited. Returns True if the datagram was handed to the OS."""
        now = self.clock()
        if self.last_send_ns is not None and now - self.last_send_ns < self.min_interval_ns:
            self.rate_limited += 1
            return False
        data = encode_landmarks(self.seq + 1, source, t_sync_ns, now, points, vis)
        try:
            self.sock.sendto(data, self.addr)
        except OSError as exc:  # includes BlockingIOError and Windows ConnectionResetError
            self.send_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.seq += 1
        self.sent += 1
        self.first_send_ns = now if self.first_send_ns is None else self.first_send_ns
        self.last_send_ns = now
        return True

    def measured_rate_hz(self) -> float | None:
        """Mean send rate over the first..last successful send, from the sender's clock."""
        if self.sent < 2 or self.last_send_ns == self.first_send_ns:
            return None
        return (self.sent - 1) / ((self.last_send_ns - self.first_send_ns) * 1e-9)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
