"""Headless smoke check of the raw camera shadow scene + live add-on, run INSIDE Blender 5.0
(bpy + stdlib only).

  blender --background --factory-startup <scene.blend> --python-exit-code 1
      --python blender/tests/smoke_check.py -- --offset X Y Z --landmarks nose l_shoulder ...
      --cam DISTANCE HEIGHT YAW HFOV --resolution W H

Driven by tests/test_blender_smoke.py. Any failed check raises, so Blender exits with code 1. Prints
'EKAGRATA SMOKE OK' on success.
"""

import argparse
import math
import socket
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
import ekagrata_live  # noqa: E402
from ekagrata_live import core  # noqa: E402

TOL_M = 1e-6


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def close(a, b, tol=TOL_M):
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--offset", type=float, nargs=3, required=True)
    p.add_argument("--landmarks", nargs="+", required=True)
    p.add_argument("--cam", type=float, nargs=4, required=True, metavar=("DIST", "HEIGHT", "YAW", "HFOV"))
    p.add_argument("--resolution", type=int, nargs=2, required=True)
    return p.parse_args(argv)


def check_scene(landmarks):
    objs = bpy.data.objects
    expected = ["floor", "net", "sun_key", "sun_fill", "camera", "cam_matched", "shadow_root", "shadow_label",
                "line_baseline_neg", "line_baseline_pos", "line_side_left", "line_side_right",
                "line_short_service_neg", "line_short_service_pos", "line_centre_neg", "line_centre_pos"]
    expected += [core.empty_name(n) for n in landmarks]
    expected += [core.link_name(a, b) for a, b in core.links_for(landmarks)]
    missing = [n for n in expected if n not in objs]
    check(not missing, f"all {len(expected)} expected objects exist (missing: {missing})")
    check(all(objs[core.empty_name(n)].type == "EMPTY" for n in landmarks), "landmarks are Empties")
    check(bpy.context.scene.camera is objs["cam_matched"], "scene camera is cam_matched")
    xs = [v.co.x for v in objs["line_side_left"].data.vertices]
    check(math.isclose(min(xs), -6.70, abs_tol=1e-6) and math.isclose(max(xs), 6.70, abs_tol=1e-6),
          "side line spans the 13.40 m court length")
    ys = [v.co.y for v in objs["line_baseline_pos"].data.vertices]
    check(math.isclose(min(ys), -2.59, abs_tol=1e-6) and math.isclose(max(ys), 2.59, abs_tol=1e-6),
          "baseline spans the 5.18 m singles width")
    zs = [v.co.z for v in objs["net"].data.vertices]
    check(math.isclose(max(zs), 1.524, abs_tol=1e-6), "net top at 1.524 m")
    check("raw camera shadow" in objs["shadow_label"].data.body and core.AXIS_CAVEAT in
          objs["shadow_label"].data.body, "label says raw camera shadow + axis caveat")
    hit = [o.name for o in objs if o.name.startswith("link_") and o.active_material.name == "mat_hitting"]
    check(sorted(hit) == sorted(core.link_name(a, b) for a, b in core.links_for(landmarks)
                                if core.is_hitting_arm(a, b, "right")), f"hitting-arm links coloured: {hit}")


def check_matched_camera(offset, landmarks, cam_args, resolution):
    """cam_matched: expected pose, looks along the expected direction, horizontal FOV, sees the figure, and
    the person's right side is on screen-left (no mirroring)."""
    scene = bpy.context.scene
    obj = bpy.data.objects["cam_matched"]
    dist, height, yaw, hfov = cam_args
    location, forward, _ = core.matched_camera_pose(offset, dist, height, yaw)
    bpy.context.view_layer.update()
    m = obj.matrix_world
    check(close(m.translation, location), f"cam_matched at {tuple(round(c, 4) for c in location)}")
    look = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    check(close(look, forward, 1e-6), f"cam_matched looks along {tuple(round(c, 4) for c in forward)}")
    check(abs(look.z) < 1e-9, "cam_matched has zero pitch")
    check(obj.data.sensor_fit == "HORIZONTAL" and math.isclose(obj.data.angle_x, math.radians(hfov),
                                                               abs_tol=1e-6),
          f"cam_matched horizontal FOV {hfov} deg")
    check((scene.render.resolution_x, scene.render.resolution_y) == tuple(resolution),
          f"render resolution {resolution[0]}x{resolution[1]} (real camera aspect)")
    view = "matched view" if core.is_matched_view(yaw) else "alternative viewpoint"
    check(obj["ekagrata_view"] == view, f"cam_matched labelled '{view}'")
    uv = {}
    for n in landmarks:
        p = world_to_camera_view(scene, obj, bpy.data.objects[core.empty_name(n)].matrix_world.translation)
        uv[n] = p
        check(0.0 < p.x < 1.0 and 0.0 < p.y < 1.0 and p.z > 0.0, f"{n} inside the cam_matched frame")
    if core.is_matched_view(yaw):
        check(uv["r_shoulder"].x < uv["l_shoulder"].x, "right shoulder on screen-left (no mirroring)")


def check_apply(offset, landmarks):
    objects = ekagrata_live.find_shadow_objects(bpy.context.scene)
    check(len(objects["empties"]) == len(landmarks), "add-on finds every landmark Empty")
    # Known message: distinct, non-placeholder points; l_hip low visibility, nose missing entirely.
    points = {n: [0.1 * i, -0.05 * i, 0.02 * i + 0.3] for i, n in enumerate(landmarks) if n != "nose"}
    vis = {n: 0.9 for n in points}
    vis["l_hip"] = 0.2
    msg = {"v": 1, "seq": 1, "mode": "landmarks", "source": "real", "t_sync_ns": 0, "t_send_ns": 1,
           "points": points, "vis": vis}
    ekagrata_live.apply_message(objects, msg)
    bpy.context.view_layer.update()
    off = Vector(offset)
    for n in landmarks:
        obj = bpy.data.objects[core.empty_name(n)]
        if n in ("nose", "l_hip"):
            check(obj.hide_get(), f"{n} hidden (missing or vis < 0.5)")
        else:
            check(not obj.hide_get() and close(obj.matrix_world.translation, off + Vector(points[n])),
                  f"{n} at offset + point")
    for a, b in core.links_for(landmarks):
        obj = bpy.data.objects[core.link_name(a, b)]
        if "l_hip" in (a, b):
            check(obj.hide_get(), f"link {a}-{b} hidden with its landmark")
            continue
        m = obj.matrix_world
        ends = (m @ Vector((0, 0, -0.5)), m @ Vector((0, 0, 0.5)))
        check(close(ends[0], off + Vector(points[a]), 1e-5) and close(ends[1], off + Vector(points[b]), 1e-5),
              f"link {a}-{b} spans its two landmarks")
    msg2 = dict(msg, seq=2, t_send_ns=2, vis=dict(vis, l_hip=0.8))
    ekagrata_live.apply_message(objects, msg2)
    check(not bpy.data.objects[core.empty_name("l_hip")].hide_get(), "l_hip shown again when vis >= 0.5")


def check_udp_keep_newest():
    """Real loopback socket inside Blender's Python: drain three datagrams, apply only the newest."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for seq in (1, 2, 3):
        body = (f'{{"v":1,"seq":{seq},"mode":"landmarks","source":"replay","t_sync_ns":{seq},'
                f'"t_send_ns":{seq * 10},"points":{{}},"vis":{{}}}}')
        tx.sendto(body.encode("utf-8"), rx.getsockname())
    tx.sendto(b"not json", rx.getsockname())
    time.sleep(0.05)
    msgs, bad = ekagrata_live._drain(rx)
    rx.close()
    tx.close()
    stats = core.LinkStats()
    newest = core.process_batch(stats, msgs, bad, time.perf_counter_ns())
    check(len(msgs) == 3 and bad == 1, "drained 3 valid + 1 bad datagram from a real socket")
    check(newest["seq"] == 3 and stats.superseded == 2 and stats.applied == 1, "only the newest applied")


def main():
    args = parse_args()
    print(f"[EKAGRATA smoke] Blender {bpy.app.version_string}, Python {sys.version.split()[0]}")
    check_scene(args.landmarks)
    check_matched_camera(args.offset, args.landmarks, args.cam, args.resolution)
    ekagrata_live.register()
    try:
        check(hasattr(bpy.types, "EKAGRATA_PT_live") and hasattr(bpy.ops.ekagrata, "live_start"),
              "add-on panel and operators registered")
        check(ekagrata_live._on_load_pre in bpy.app.handlers.load_pre, "load_pre handler installed")
        check_apply(args.offset, args.landmarks)
        check_udp_keep_newest()
    finally:
        ekagrata_live.unregister()
    check(not hasattr(bpy.types, "EKAGRATA_PT_live"), "add-on unregistered cleanly")
    print("EKAGRATA SMOKE OK")


main()
