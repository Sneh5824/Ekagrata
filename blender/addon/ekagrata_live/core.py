"""Pure-Python logic of the EKAGRATA raw camera shadow receiver (no bpy; Python 3.11 standard library only).

Tested outside Blender by tests/test_blender_live_core.py. The raw camera shadow is NOT the Digital Twin:
Empties are placed at raw landmark positions, with no fixed segment lengths and no joint rotations.

Clock note: ages are `time.perf_counter_ns()` in Blender minus a timestamp from the sender process. On
Windows, perf_counter_ns() is QueryPerformanceCounter, which is one system-wide clock, so the difference is
valid across processes ON THE SAME MACHINE. It is meaningless across machines.
"""

import json
import math
from collections import deque
from dataclasses import dataclass, field

MSG_VERSION = 1
LATENCY_WINDOW_NS = 2_000_000_000  # median/p95 and rates are computed over the last 2 s
RUN_HISTORY_MAX = 200_000  # whole-run latency history kept for the Stop summary (~1.8 h at 30 Hz)
VIS_THRESHOLD = 0.5  # landmarks with visibility below this (or missing) are hidden
AXIS_CAVEAT = "axis mapping unverified until check_axes passes"
REPLAY_CAPTURE_NOTE = "n/a in replay (t_sync_ns is from the recording)"

# Stick-figure links (landmark name pairs). Hand = wrist-index, wrist-pinky, index-pinky.
_SIDE_LINKS = (("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "index"), ("wrist", "pinky"),
               ("index", "pinky"), ("shoulder", "hip"))
LINKS = tuple((f"{s}_{a}", f"{s}_{b}") for s in ("l", "r") for a, b in _SIDE_LINKS) + (
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"))
HITTING_ARM_PARTS = ("shoulder", "elbow", "wrist", "index", "pinky")


def empty_name(landmark: str) -> str:
    return f"lm_{landmark}"


def link_name(a: str, b: str) -> str:
    return f"link_{a}__{b}"


def parse_link_name(name: str):
    """Inverse of link_name; None if `name` is not a link object name."""
    if not name.startswith("link_") or "__" not in name:
        return None
    a, b = name[len("link_"):].split("__", 1)
    return (a, b) if a and b else None


def links_for(landmarks) -> list:
    """LINKS whose two ends are both in `landmarks`."""
    have = set(landmarks)
    return [(a, b) for a, b in LINKS if a in have and b in have]


def is_hitting_arm(a: str, b: str, hitting_side: str) -> bool:
    """True for a link between two landmarks of the hitting arm/hand (shoulder-hip is torso, not arm)."""
    prefix = "r_" if hitting_side == "right" else "l_"
    return all(n.startswith(prefix) and n[2:] in HITTING_ARM_PARTS for n in (a, b))


def decode_message(data: bytes):
    """Parse one datagram. Returns the message dict, or None if it is not a valid v1 landmarks message."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != MSG_VERSION or msg.get("mode") != "landmarks":
        return None
    for key in ("seq", "t_sync_ns", "t_send_ns"):
        if not isinstance(msg.get(key), int) or isinstance(msg.get(key), bool):
            return None
    points, vis = msg.get("points"), msg.get("vis")
    if not isinstance(points, dict) or not isinstance(vis, dict):
        return None
    for p in points.values():
        if not (isinstance(p, list) and len(p) == 3 and all(isinstance(c, (int, float)) for c in p)
                and all(math.isfinite(c) for c in p)):
            return None
    return msg


def visible_landmarks(msg: dict, names, threshold: float = VIS_THRESHOLD) -> dict:
    """{name: bool} for every configured landmark: visible only if present with vis >= threshold."""
    points, vis = msg["points"], msg["vis"]
    out = {}
    for n in names:
        v = vis.get(n)
        out[n] = n in points and isinstance(v, (int, float)) and v >= threshold
    return out


def percentile(values, q: float):
    """Nearest-rank percentile (q in [0, 100]) of a non-empty sequence; None if empty."""
    s = sorted(values)
    if not s:
        return None
    k = max(1, math.ceil(q / 100.0 * len(s)))
    return s[k - 1]


def median(values):
    s = sorted(values)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


@dataclass
class LinkStats:
    """Receiver counters and 2 s sliding windows. All times are perf_counter_ns() of the receiver."""

    received: int = 0  # valid messages read from the socket
    applied: int = 0  # messages applied to the scene (one per tick at most)
    superseded: int = 0  # valid, newer than the last applied, but a newer one arrived in the same tick
    stale: int = 0  # not newer (t_send_ns) than the last applied message: ignored
    lost: int = 0  # gaps in seq among accepted messages (a late, reordered packet counts here, then as stale)
    bad: int = 0  # datagrams that were not valid v1 landmarks messages
    restarts: int = 0  # seq went backwards while t_send_ns went forwards (sender restarted)
    last_seq: int | None = None
    last_t_send_ns: int | None = None
    last_rx_ns: int | None = None
    last_source: str = ""
    rx_times: deque = field(default_factory=deque)  # receive time per valid message
    applies: deque = field(default_factory=deque)  # (t_apply_ns, age_send_ns, age_capture_ns | None)
    run_ages: deque = field(default_factory=lambda: deque(maxlen=RUN_HISTORY_MAX))  # same, whole run


def process_batch(stats: LinkStats, msgs: list, n_bad: int, now_ns: int):
    """Account for one tick's drained messages and return the ONE newest message to apply (or None).

    Newest = largest t_send_ns (not seq, so a restarted sender that resets seq is not treated as stale).
    Records age since send and, for live data (source "real"), age since capture."""
    stats.bad += n_bad
    stats.received += len(msgs)
    for _ in msgs:
        stats.rx_times.append(now_ns)
    if msgs:
        stats.last_rx_ns = now_ns
    last = stats.last_t_send_ns
    fresh = sorted((m for m in msgs if last is None or m["t_send_ns"] > last), key=lambda m: m["t_send_ns"])
    stats.stale += len(msgs) - len(fresh)
    for m in fresh:
        if stats.last_seq is not None:
            if m["seq"] > stats.last_seq + 1:
                stats.lost += m["seq"] - stats.last_seq - 1
            elif m["seq"] <= stats.last_seq:
                stats.restarts += 1
        stats.last_seq = m["seq"]
    _prune(stats, now_ns)
    if not fresh:
        return None
    newest = fresh[-1]
    stats.superseded += len(fresh) - 1
    stats.applied += 1
    stats.last_t_send_ns = newest["t_send_ns"]
    stats.last_source = newest.get("source", "")
    age_capture = now_ns - newest["t_sync_ns"] if newest.get("source") == "real" else None
    stats.applies.append((now_ns, now_ns - newest["t_send_ns"], age_capture))
    stats.run_ages.append(stats.applies[-1])
    return newest


def _prune(stats: LinkStats, now_ns: int) -> None:
    cutoff = now_ns - LATENCY_WINDOW_NS
    while stats.rx_times and stats.rx_times[0] < cutoff:
        stats.rx_times.popleft()
    while stats.applies and stats.applies[0][0] < cutoff:
        stats.applies.popleft()


def window_summary(stats: LinkStats, now_ns: int) -> dict:
    """Rates (Hz) and latency median/p95 (ms) over the last 2 s. Values are None when there is no data;
    capture-age values are None in replay mode (t_sync_ns is from the recording, not this clock)."""
    _prune(stats, now_ns)
    span_s = LATENCY_WINDOW_NS * 1e-9
    send = [a[1] * 1e-6 for a in stats.applies]
    cap = [a[2] * 1e-6 for a in stats.applies if a[2] is not None]
    return {
        "rx_hz": len(stats.rx_times) / span_s,
        "applied_hz": len(stats.applies) / span_s,
        "age_send_ms_median": median(send),
        "age_send_ms_p95": percentile(send, 95),
        "age_capture_ms_median": median(cap),
        "age_capture_ms_p95": percentile(cap, 95),
        "since_last_rx_ms": None if stats.last_rx_ns is None else (now_ns - stats.last_rx_ns) * 1e-6,
        "source": stats.last_source,
    }


def run_summary(stats: LinkStats) -> dict:
    """Latency median/p95 (ms) over the whole run (last RUN_HISTORY_MAX applied messages)."""
    send = [a[1] * 1e-6 for a in stats.run_ages]
    cap = [a[2] * 1e-6 for a in stats.run_ages if a[2] is not None]
    return {"n": len(send), "age_send_ms_median": median(send), "age_send_ms_p95": percentile(send, 95),
            "age_capture_ms_median": median(cap), "age_capture_ms_p95": percentile(cap, 95)}


def link_transform(a, b):
    """Placement of a unit-height cylinder (along its local +Z, centred at its origin) spanning a -> b.

    Returns (midpoint (3,), quaternion [w, x, y, z] rotating +Z onto b - a, length). A zero-length link
    returns the identity rotation and length 0."""
    d = [b[i] - a[i] for i in range(3)]
    mid = [(a[i] + b[i]) / 2.0 for i in range(3)]
    length = math.sqrt(sum(c * c for c in d))
    if length < 1e-12:
        return mid, [1.0, 0.0, 0.0, 0.0], 0.0
    u = [c / length for c in d]
    # q = normalise(1 + z.u, z x u) with z = (0, 0, 1): z x u = (-u_y, u_x, 0)
    w = 1.0 + u[2]
    if w < 1e-9:  # u == -z: any half turn about an axis perpendicular to z
        return mid, [0.0, 1.0, 0.0, 0.0], length
    q = [w, -u[1], u[0], 0.0]
    n = math.sqrt(sum(c * c for c in q))
    return mid, [c / n for c in q], length
