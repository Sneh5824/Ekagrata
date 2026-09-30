"""Pure-Python logic of the Blender add-on (blender/addon/ekagrata_live/core.py), tested outside Blender."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ekagrata.core import quat
from ekagrata.core.config import load_blender_config

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ekagrata_live_core",
                                               REPO / "blender" / "addon" / "ekagrata_live" / "core.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

S = 1_000_000_000


def msg(seq, t_send_ns, source="real", t_sync_ns=0, points=None, vis=None):
    return {"v": 1, "seq": seq, "mode": "landmarks", "source": source, "t_sync_ns": t_sync_ns,
            "t_send_ns": t_send_ns, "points": points or {}, "vis": vis or {}}


def raw(m):
    return json.dumps(m).encode("utf-8")


def test_decode_message_valid_and_invalid():
    m = msg(3, 10, points={"a": [1, 2.5, -3]}, vis={"a": 0.7})
    assert core.decode_message(raw(m)) == m
    assert core.decode_message(b"\xff\xfe") is None
    assert core.decode_message(b"not json") is None
    assert core.decode_message(raw(dict(m, v=2))) is None
    assert core.decode_message(raw(dict(m, seq=True))) is None
    assert core.decode_message(raw(dict(m, points={"a": [1, 2]}))) is None
    assert core.decode_message(b'{"v":1,"seq":1,"mode":"landmarks","t_sync_ns":0,"t_send_ns":0,'
                               b'"points":{"a":[NaN,0,0]},"vis":{}}') is None


def test_keep_newest_in_one_tick():
    st = core.LinkStats()
    newest = core.process_batch(st, [msg(2, 20), msg(1, 10), msg(3, 30)], 1, 100)
    assert newest["seq"] == 3
    assert (st.received, st.applied, st.superseded, st.stale, st.lost, st.bad) == (3, 1, 2, 0, 0, 1)


def test_duplicates_and_old_messages_are_stale():
    st = core.LinkStats()
    core.process_batch(st, [msg(3, 30)], 0, 100)
    assert core.process_batch(st, [msg(3, 30), msg(2, 20)], 0, 200) is None
    assert st.stale == 2 and st.applied == 1


def test_seq_gap_counts_lost():
    st = core.LinkStats()
    core.process_batch(st, [msg(1, 10)], 0, 100)
    core.process_batch(st, [msg(5, 50)], 0, 200)
    assert st.lost == 3


def test_sender_restart_is_not_stale():
    st = core.LinkStats()
    core.process_batch(st, [msg(500, 10 * S)], 0, 100)
    newest = core.process_batch(st, [msg(1, 20 * S)], 0, 200)  # seq reset, later send time
    assert newest["seq"] == 1 and st.restarts == 1 and st.stale == 0 and st.lost == 0


def test_no_messages_applies_nothing():
    st = core.LinkStats()
    assert core.process_batch(st, [], 0, 100) is None
    assert st.applied == 0 and st.last_rx_ns is None


def test_latency_windows_live_and_replay():
    st = core.LinkStats()
    # live: sent 1..5 ms before apply, captured 30 ms before apply
    for i in range(1, 6):
        now = i * 100_000_000
        core.process_batch(st, [msg(i, now - i * 1_000_000, "real", t_sync_ns=now - 30_000_000)], 0, now)
    w = core.window_summary(st, 500_000_000)
    assert w["age_send_ms_median"] == pytest.approx(3.0)
    assert w["age_send_ms_p95"] == pytest.approx(5.0)
    assert w["age_capture_ms_median"] == pytest.approx(30.0)
    assert w["rx_hz"] == pytest.approx(2.5) and w["applied_hz"] == pytest.approx(2.5)
    # replay: capture age is not meaningful and is not recorded
    core.process_batch(st, [msg(6, 590_000_000, "replay", t_sync_ns=0)], 0, 600_000_000)
    assert st.applies[-1][2] is None
    assert core.window_summary(st, 600_000_000)["source"] == "replay"


def test_window_drops_entries_older_than_2_s():
    st = core.LinkStats()
    core.process_batch(st, [msg(1, 0)], 0, 1)
    w = core.window_summary(st, 1 + 2 * S + 1)
    assert w["rx_hz"] == 0 and w["age_send_ms_median"] is None
    assert w["since_last_rx_ms"] == pytest.approx(2000.0, abs=1e-3)


def test_percentile_and_median_known_answers():
    assert core.percentile(range(1, 21), 95) == 19
    assert core.percentile([5.0], 95) == 5.0
    assert core.percentile([], 95) is None
    assert core.median([3, 1, 2]) == 2 and core.median([4, 1, 2, 3]) == 2.5 and core.median([]) is None


def test_visible_landmarks_threshold():
    m = msg(1, 1, points={"a": [0, 0, 0], "b": [0, 0, 0], "c": [0, 0, 0]}, vis={"a": 0.5, "b": 0.49})
    assert core.visible_landmarks(m, ["a", "b", "c", "d"]) == {"a": True, "b": False, "c": False, "d": False}


@pytest.mark.parametrize("seed", range(20))
def test_link_transform_spans_the_endpoints(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=3), rng.normal(size=3)
    mid, q, length = core.link_transform(a.tolist(), b.tolist())
    assert length == pytest.approx(np.linalg.norm(b - a))
    assert np.linalg.norm(q) == pytest.approx(1.0)
    half = quat.rotate_vector(np.array(q), np.array([0.0, 0.0, length / 2]))
    assert np.allclose(np.array(mid) - half, a, atol=1e-12)
    assert np.allclose(np.array(mid) + half, b, atol=1e-12)


def test_link_transform_edge_cases():
    mid, q, length = core.link_transform([0, 0, 1], [0, 0, 0])  # antiparallel to +Z
    assert length == 1.0 and np.allclose(quat.rotate_vector(np.array(q), [0, 0, 1]), [0, 0, -1])
    mid, q, length = core.link_transform([0, 0, 0], [0, 0, 2])  # parallel to +Z
    assert np.allclose(q, [1, 0, 0, 0]) and mid == [0, 0, 1]
    mid, q, length = core.link_transform([1, 2, 3], [1, 2, 3])  # zero length
    assert length == 0.0 and q == [1.0, 0.0, 0.0, 0.0] and mid == [1, 2, 3]


def test_links_names_and_hitting_arm():
    landmarks = load_blender_config(REPO / "configs" / "blender.yaml").landmarks
    links = core.links_for(landmarks)
    assert len(links) == 14 and set(links) == set(core.LINKS)
    for a, b in links:
        assert core.parse_link_name(core.link_name(a, b)) == (a, b)
    right = {(a, b) for a, b in links if core.is_hitting_arm(a, b, "right")}
    assert right == {("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("r_wrist", "r_index"),
                     ("r_wrist", "r_pinky"), ("r_index", "r_pinky")}
    left = {(a, b) for a, b in links if core.is_hitting_arm(a, b, "left")}
    assert all(a.startswith("l_") for a, _ in left) and len(left) == 5
    assert core.links_for(["r_shoulder", "r_elbow"]) == [("r_shoulder", "r_elbow")]
    assert core.parse_link_name("lm_nose") is None and core.parse_link_name("link_a") is None


def test_run_summary_keeps_whole_run_after_window_expires():
    st = core.LinkStats()
    for i in range(1, 5):  # age since send 1..4 ms, capture 10 ms
        now = i * 100_000_000
        core.process_batch(st, [msg(i, now - i * 1_000_000, "real", t_sync_ns=now - 10_000_000)], 0, now)
    assert core.window_summary(st, 10 * S)["age_send_ms_median"] is None  # 2 s window expired
    run = core.run_summary(st)
    assert run["n"] == 4 and run["age_send_ms_median"] == pytest.approx(2.5)
    assert run["age_send_ms_p95"] == pytest.approx(4.0)
    assert run["age_capture_ms_median"] == pytest.approx(10.0)
