"""EKAGRATA live link: drives the Blender RAW CAMERA SHADOW from UDP landmark messages (M5-preview).

NOT the Digital Twin: Empties are placed at raw MediaPipe landmark positions (no fixed segment lengths, no
joint rotations). Blender 5.0 Python API; bpy + standard library only.
Scene: blender/scripts/build_preview_scene.py.

Per tick (~60 Hz modal TIMER): drain the non-blocking socket, apply ONLY the newest message, set locations /
transforms only; visibility is changed only when a landmark's visible state changes; no object is created or
deleted in the modal loop. Latency clock note: see core.py (QueryPerformanceCounter is system-wide on
Windows).
"""

import socket
import time

import bpy
from bpy.app.handlers import persistent

from . import core

bl_info = {
    "name": "EKAGRATA live (raw camera shadow)",
    "author": "EKAGRATA",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > EKAGRATA",
    "description": "UDP receiver that drives the EKAGRATA raw camera shadow (not the Digital Twin)",
    "category": "3D View",
}

BIND_HOST = "127.0.0.1"  # loopback only, never 0.0.0.0
TICK_S = 1.0 / 60.0
MAX_DRAIN = 10000  # datagrams read per tick at most
REDRAW_INTERVAL_NS = 200_000_000


class _Live:
    """Module-level receiver state (one receiver per Blender session)."""

    sock = None
    timer = None
    running = False
    stats = None
    objects = None  # {"empties": {name: obj}, "links": [(a, b, obj)], "visible": {name: bool}}
    last_redraw_ns = 0
    error = ""


def find_shadow_objects(scene) -> dict:
    """Look up the shadow's objects once (at Start): Empties 'lm_<name>' and cylinders 'link_<a>__<b>'."""
    empties, links = {}, []
    for obj in scene.objects:
        if obj.name.startswith("lm_") and obj.type == "EMPTY":
            empties[obj.name[len("lm_"):]] = obj
        else:
            pair = core.parse_link_name(obj.name)
            if pair is not None:
                links.append((pair[0], pair[1], obj))
    return {"empties": empties, "links": links, "visible": {}}


def _set_visible(obj, visible: bool) -> None:
    obj.hide_set(not visible)
    obj.hide_render = not visible


def apply_message(objects: dict, msg: dict) -> None:
    """Place Empties/links from one message (positions are relative to the 'shadow_root' parent)."""
    empties = objects["empties"]
    shown = core.visible_landmarks(msg, empties.keys())
    cache = objects["visible"]
    points = msg["points"]
    for name, obj in empties.items():
        if shown[name]:
            obj.location = points[name]
        if cache.get(name) != shown[name]:
            _set_visible(obj, shown[name])
            cache[name] = shown[name]
    for a, b, obj in objects["links"]:
        both = shown.get(a, False) and shown.get(b, False)
        key = f"{a}__{b}"
        if both:
            mid, q, length = core.link_transform(points[a], points[b])
            obj.location = mid
            obj.rotation_quaternion = q
            obj.scale = (1.0, 1.0, max(length, 1e-6))
        if cache.get(key) != both:
            _set_visible(obj, both)
            cache[key] = both


def _drain(sock) -> tuple:
    msgs, bad = [], 0
    for _ in range(MAX_DRAIN):
        try:
            data = sock.recv(65535)
        except BlockingIOError:
            break
        except ConnectionResetError:  # Windows reports ICMP port-unreachable on UDP; harmless here
            continue
        except OSError:
            break
        m = core.decode_message(data)
        if m is None:
            bad += 1
        else:
            msgs.append(m)
    return msgs, bad


def _fmt(v, spec=".1f"):
    return "n/a" if v is None else format(v, spec)


def _age_lines(label, w, source) -> list:
    cap = (f"median {_fmt(w['age_capture_ms_median'])} / p95 {_fmt(w['age_capture_ms_p95'])} ms"
           if source != "replay" else core.REPLAY_CAPTURE_NOTE)
    return [f"{label} age since send: median {_fmt(w['age_send_ms_median'])} / "
            f"p95 {_fmt(w['age_send_ms_p95'])} ms",
            f"{label} age since capture: {cap}"]


def _summary_lines(stats, now_ns) -> list:
    w = core.window_summary(stats, now_ns)
    run = core.run_summary(stats)
    return [
        f"received {stats.received}, applied {stats.applied}, superseded {stats.superseded}, "
        f"stale {stats.stale}, lost (seq gaps) {stats.lost}, bad {stats.bad}, restarts {stats.restarts}",
        f"last 2 s: rx {w['rx_hz']:.1f} Hz, applied {w['applied_hz']:.1f} Hz",
        *_age_lines("last 2 s:", w, stats.last_source),
        *_age_lines(f"whole run ({run['n']} applied):", run, stats.last_source),
    ]


def stop_live(reason: str) -> None:
    """Close the socket and remove the timer. Safe to call repeatedly."""
    was_running = _Live.running
    _Live.running = False
    if _Live.timer is not None:
        try:
            bpy.context.window_manager.event_timer_remove(_Live.timer)
        except (ReferenceError, AttributeError, RuntimeError):
            pass
        _Live.timer = None
    if _Live.sock is not None:
        try:
            _Live.sock.close()
        except OSError:
            pass
        _Live.sock = None
    if was_running and _Live.stats is not None:
        print(f"[EKAGRATA raw camera shadow] stopped ({reason}); {core.AXIS_CAVEAT}")
        for line in _summary_lines(_Live.stats, time.perf_counter_ns()):
            print("  " + line)


class EKAGRATA_OT_live_start(bpy.types.Operator):
    """Start receiving landmark messages and drive the raw camera shadow"""

    bl_idname = "ekagrata.live_start"
    bl_label = "Start"

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        if _Live.running:
            self.report({"WARNING"}, "EKAGRATA live link already running")
            return {"CANCELLED"}
        port = context.window_manager.ekagrata_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((BIND_HOST, port))
        except OSError as exc:
            sock.close()
            _Live.error = f"cannot bind {BIND_HOST}:{port}: {exc}"
            self.report({"ERROR"}, _Live.error)
            return {"CANCELLED"}
        sock.setblocking(False)
        _Live.sock, _Live.error = sock, ""
        _Live.stats = core.LinkStats()
        _Live.objects = find_shadow_objects(context.scene)
        if not _Live.objects["empties"]:
            self.report({"WARNING"}, "no 'lm_*' Empties in this scene: build it with build_preview_scene.py")
        _Live.running = True
        wm = context.window_manager
        _Live.timer = wm.event_timer_add(TICK_S, window=context.window)
        wm.modal_handler_add(self)
        print(f"[EKAGRATA raw camera shadow] listening on {BIND_HOST}:{port}; {core.AXIS_CAVEAT}")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not _Live.running:
            return {"FINISHED"}
        if event.type != "TIMER" or _Live.sock is None:
            return {"PASS_THROUGH"}
        msgs, bad = _drain(_Live.sock)
        now = time.perf_counter_ns()
        newest = core.process_batch(_Live.stats, msgs, bad, now)
        if newest is not None:
            apply_message(_Live.objects, newest)
        if now - _Live.last_redraw_ns > REDRAW_INTERVAL_NS:
            _Live.last_redraw_ns = now
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        return {"PASS_THROUGH"}


class EKAGRATA_OT_live_stop(bpy.types.Operator):
    """Stop the live link and close the socket"""

    bl_idname = "ekagrata.live_stop"
    bl_label = "Stop"

    def execute(self, context):
        stop_live("Stop pressed")
        return {"FINISHED"}


class EKAGRATA_PT_live(bpy.types.Panel):
    bl_label = "Raw camera shadow"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "EKAGRATA"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Raw camera shadow (NOT the Digital Twin)")
        col.label(text=core.AXIS_CAVEAT, icon="ERROR")
        row = layout.row(align=True)
        row.enabled = not _Live.running
        row.prop(context.window_manager, "ekagrata_port")
        row = layout.row(align=True)
        if _Live.running:
            row.operator(EKAGRATA_OT_live_stop.bl_idname, icon="PAUSE")
        else:
            row.operator(EKAGRATA_OT_live_start.bl_idname, icon="PLAY")
        if _Live.error:
            layout.label(text=_Live.error, icon="ERROR")
        s = _Live.stats
        if s is None:
            return
        w = core.window_summary(s, time.perf_counter_ns())
        box = layout.box().column(align=True)
        box.label(text=f"Receive rate: {w['rx_hz']:.1f} Hz (applied {w['applied_hz']:.1f} Hz)")
        box.label(text=f"Since last packet: {_fmt(w['since_last_rx_ms'], '.0f')} ms")
        box.label(text="Age since send (last 2 s):")
        box.label(text=f"  median {_fmt(w['age_send_ms_median'])} ms, p95 {_fmt(w['age_send_ms_p95'])} ms")
        box.label(text="Age since capture (last 2 s):")
        if w["source"] == "replay":
            box.label(text="  " + core.REPLAY_CAPTURE_NOTE)
        else:
            box.label(text=f"  median {_fmt(w['age_capture_ms_median'])} ms, "
                           f"p95 {_fmt(w['age_capture_ms_p95'])} ms")
        box.label(text=f"Source: {w['source'] or 'n/a'}")
        box.label(text=f"Received {s.received}, applied {s.applied}")
        box.label(text=f"Superseded {s.superseded}, stale {s.stale}, lost {s.lost}")
        box.label(text=f"Bad {s.bad}, sender restarts {s.restarts}")


@persistent
def _on_load_pre(_filepath):
    stop_live("file load")


_CLASSES = (EKAGRATA_OT_live_start, EKAGRATA_OT_live_stop, EKAGRATA_PT_live)


def register():
    bpy.types.WindowManager.ekagrata_port = bpy.props.IntProperty(
        name="Port", description="UDP port on 127.0.0.1", default=9870, min=1024, max=65535)
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)


def unregister():
    stop_live("add-on unregistered")
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.WindowManager.ekagrata_port
