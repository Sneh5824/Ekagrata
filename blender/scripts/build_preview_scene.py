"""Build the Blender scene for the EKAGRATA RAW CAMERA SHADOW (M5-preview; NOT the Digital Twin).

Run with Blender 5.0 (bpy + standard library only), normally via scripts/build_blender_preview.py:
  blender --background --factory-startup --python-exit-code 1
      --python blender/scripts/build_preview_scene.py --
      --out blender/ekagrata_preview.blend --offset -4.0 0.0 1.0 --hitting-side right
      --landmarks nose l_shoulder ... --cam-distance 2.5 --cam-height 0.3 --cam-yaw 0 --cam-hfov 70
      --resolution 1280 720

Scene (EKAGRATA world: X toward the net, Y left, Z up, metres): floor, singles court lines (13.40 x 5.18 m),
net (1.524 m at centre), lights, cameras ('cam_matched' = scene camera, matching the real camera for
the side-by-side view; 'camera' = three-quarter overview), one Empty 'lm_<name>' per landmark and one unit
cylinder
'link_<a>__<b>' per stick-figure link, all parented to 'shadow_root' at the placement offset. The Empties
start in a PLACEHOLDER pose (not data) until the live link applies a message.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ekagrata_live import core  # noqa: E402  (after sys.path setup; core.py is stdlib-only)

COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18
DOUBLES_WIDTH_M = 6.10  # net / posts span the doubles width
SHORT_SERVICE_FROM_NET_M = 1.98
LINE_WIDTH_M = 0.04
NET_HEIGHT_CENTRE_M = 1.524
NET_DEPTH_M = 0.76
LINK_RADIUS_M = 0.015
EMPTY_SIZE_M = 0.035
LABEL = f"EKAGRATA raw camera shadow - NOT the Digital Twin\n{core.AXIS_CAVEAT}"

# PLACEHOLDER pose (metres, relative to the hip midpoint, facing +X) so the figure is visible before data
# arrives. It is NOT measured data.
PLACEHOLDER = {
    "nose": (0.08, 0.0, 0.68),
    "l_shoulder": (0.0, 0.18, 0.50), "r_shoulder": (0.0, -0.18, 0.50),
    "l_elbow": (0.0, 0.21, 0.22), "r_elbow": (0.0, -0.21, 0.22),
    "l_wrist": (0.0, 0.22, -0.05), "r_wrist": (0.0, -0.22, -0.05),
    "l_index": (0.03, 0.22, -0.15), "r_index": (0.03, -0.22, -0.15),
    "l_pinky": (-0.01, 0.22, -0.13), "r_pinky": (-0.01, -0.22, -0.13),
    "l_hip": (0.0, 0.10, 0.0), "r_hip": (0.0, -0.10, 0.0),
}

COLORS = {
    "floor": (0.10, 0.28, 0.16, 1.0),
    "line": (0.95, 0.95, 0.95, 1.0),
    "net": (0.15, 0.15, 0.15, 1.0),
    "body": (0.70, 0.70, 0.72, 1.0),
    "hitting": (1.00, 0.45, 0.05, 1.0),
    "landmark": (0.20, 0.60, 1.00, 1.0),
    "label": (1.00, 0.85, 0.20, 1.0),
}


def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description="Build the EKAGRATA raw camera shadow scene")
    p.add_argument("--out", required=True, help="output .blend path")
    p.add_argument("--offset", type=float, nargs=3, default=(-4.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    p.add_argument("--hitting-side", choices=("right", "left"), default="right")
    p.add_argument("--landmarks", nargs="+", default=list(PLACEHOLDER))
    p.add_argument("--cam-distance", type=float, default=2.5, help="matched camera: hips -> lens, m")
    p.add_argument("--cam-height", type=float, default=0.3, help="matched camera: lens minus hip height, m")
    p.add_argument("--cam-yaw", type=float, default=0.0, help="0 = matched view; else alternative viewpoint")
    p.add_argument("--cam-hfov", type=float, default=70.0, help="matched camera horizontal FOV, degrees")
    p.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"),
                   help="render resolution = real camera resolution (same aspect ratio)")
    args = p.parse_args(argv)
    unknown = [n for n in args.landmarks if n not in PLACEHOLDER]
    if unknown:
        p.error(f"unknown landmark(s) {unknown}; allowed: {sorted(PLACEHOLDER)}")
    return args


def material(name, rgba):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba  # Solid-mode viewport colour
    bsdf = m.node_tree.nodes.get("Principled BSDF") if m.node_tree is not None else None
    if bsdf is not None:  # Blender 5.0 creates this node tree for new materials (use_nodes is deprecated)
        bsdf.inputs["Base Color"].default_value = rgba
    return m


def mesh_object(name, verts, faces, mat, collection):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj


def rect(x0, x1, y0, y1, z=0.0):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2, 3)]


def unit_cylinder(name, mat, segments=12):
    """Cylinder of radius LINK_RADIUS_M and height 1 along +Z, centred at the origin."""
    verts = []
    for z in (-0.5, 0.5):
        for i in range(segments):
            a = 2 * math.pi * i / segments
            verts.append((LINK_RADIUS_M * math.cos(a), LINK_RADIUS_M * math.sin(a), z))
    faces = [(i, (i + 1) % segments, segments + (i + 1) % segments, segments + i) for i in range(segments)]
    faces += [tuple(range(segments - 1, -1, -1)), tuple(range(segments, 2 * segments))]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    me.materials.append(mat)
    return me


def build_court(col, mats):
    half_l, half_w = COURT_LENGTH_M / 2, SINGLES_WIDTH_M / 2
    lw = LINE_WIDTH_M
    mesh_object("floor", *rect(-10.0, 10.0, -6.0, 6.0), mats["floor"], col)
    z = 0.002  # lines just above the floor to avoid z-fighting
    lines = {
        "line_baseline_neg": (-half_l, -half_l + lw, -half_w, half_w),
        "line_baseline_pos": (half_l - lw, half_l, -half_w, half_w),
        "line_side_left": (-half_l, half_l, half_w - lw, half_w),
        "line_side_right": (-half_l, half_l, -half_w, -half_w + lw),
        "line_short_service_neg": (-SHORT_SERVICE_FROM_NET_M - lw, -SHORT_SERVICE_FROM_NET_M,
                                   -half_w, half_w),
        "line_short_service_pos": (SHORT_SERVICE_FROM_NET_M, SHORT_SERVICE_FROM_NET_M + lw, -half_w, half_w),
        "line_centre_neg": (-half_l, -SHORT_SERVICE_FROM_NET_M, -lw / 2, lw / 2),
        "line_centre_pos": (SHORT_SERVICE_FROM_NET_M, half_l, -lw / 2, lw / 2),
    }
    for name, (x0, x1, y0, y1) in lines.items():
        mesh_object(name, *rect(x0, x1, y0, y1, z), mats["line"], col)
    half_d = DOUBLES_WIDTH_M / 2
    top, bottom = NET_HEIGHT_CENTRE_M, NET_HEIGHT_CENTRE_M - NET_DEPTH_M
    verts = [(0.0, -half_d, bottom), (0.0, half_d, bottom), (0.0, half_d, top), (0.0, -half_d, top)]
    mesh_object("net", verts, [(0, 1, 2, 3)], mats["net"], col)


def build_shadow(col, mats, offset, landmarks, hitting_side):
    root = bpy.data.objects.new("shadow_root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.2
    root.location = offset
    col.objects.link(root)
    for name in landmarks:
        e = bpy.data.objects.new(core.empty_name(name), None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = EMPTY_SIZE_M
        e.show_in_front = True
        e.parent = root
        e.location = PLACEHOLDER[name]
        e["ekagrata_placeholder"] = True
        col.objects.link(e)
    cyl = {"body": unit_cylinder("link_mesh_body", mats["body"]),
           "hitting": unit_cylinder("link_mesh_hitting", mats["hitting"])}
    for a, b in core.links_for(landmarks):
        kind = "hitting" if core.is_hitting_arm(a, b, hitting_side) else "body"
        obj = bpy.data.objects.new(core.link_name(a, b), cyl[kind])
        obj.parent = root
        obj.rotation_mode = "QUATERNION"
        mid, q, length = core.link_transform(PLACEHOLDER[a], PLACEHOLDER[b])
        obj.location = mid
        obj.rotation_quaternion = q
        obj.scale = (1.0, 1.0, max(length, 1e-6))
        col.objects.link(obj)
    return root


def build_label(col, mats, offset):
    curve = bpy.data.curves.new("shadow_label", type="FONT")
    curve.body = LABEL
    curve.align_x = "CENTER"
    curve.size = 0.09
    obj = bpy.data.objects.new("shadow_label", curve)
    obj.location = (offset[0] - 0.8, offset[1], offset[2] + 0.95)  # behind and above the figure
    obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(90.0))  # text faces +X (cam_matched)
    obj.data.materials.append(mats["label"])
    col.objects.link(obj)


def build_view(scene, col, offset):
    for name, energy, rot in (("sun_key", 3.0, (40.0, 10.0, 30.0)), ("sun_fill", 1.0, (60.0, -10.0, 210.0))):
        light = bpy.data.lights.new(name, type="SUN")
        light.energy = energy  # W/m^2 (display choice)
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = tuple(math.radians(a) for a in rot)
        col.objects.link(obj)
    cam = bpy.data.cameras.new("camera")
    cam.lens = 40.0
    cam.clip_end = 100.0
    obj = bpy.data.objects.new("camera", cam)
    # Front three-quarter view of the figure (it faces +X), aimed at a point near chest height.
    target = Vector((offset[0], offset[1], offset[2] + 0.3))
    obj.location = target + Vector((2.3, -2.3, 0.3))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (target - obj.location).to_track_quat("-Z", "Y")
    col.objects.link(obj)


def build_matched_camera(scene, col, offset, args):
    """'cam_matched': where the real camera is, looking along -X with zero pitch (see
    core.matched_camera_pose). It becomes the scene camera. No mirroring: screen right = +Y, so the person's
    right arm is on screen-left."""
    location, _forward, rows = core.matched_camera_pose(offset, args.cam_distance, args.cam_height,
                                                         args.cam_yaw)
    cam = bpy.data.cameras.new("cam_matched")
    cam.sensor_fit = "HORIZONTAL"
    cam.angle = math.radians(args.cam_hfov)  # horizontal FOV because sensor_fit is HORIZONTAL
    cam.clip_start = 0.05
    cam.clip_end = 100.0
    obj = bpy.data.objects.new("cam_matched", cam)
    obj.location = location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Matrix(rows).to_quaternion()
    matched = core.is_matched_view(args.cam_yaw)
    obj["ekagrata_view"] = "matched view" if matched else "alternative viewpoint"
    obj["ekagrata_camera"] = (f"distance {args.cam_distance} m, height {args.cam_height} m, "
                              f"yaw {args.cam_yaw} deg, horizontal FOV {args.cam_hfov} deg "
                              "(from configs/blender.yaml)")
    col.objects.link(obj)
    scene.camera = obj
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    return matched


def main():
    args = parse_args(sys.argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mats = {k: material(f"mat_{k}", v) for k, v in COLORS.items()}
    court = bpy.data.collections.new("court")
    shadow = bpy.data.collections.new("raw_camera_shadow")
    view = bpy.data.collections.new("view")
    for c in (court, shadow, view):
        scene.collection.children.link(c)
    build_court(court, mats)
    build_shadow(shadow, mats, tuple(args.offset), args.landmarks, args.hitting_side)
    build_label(shadow, mats, tuple(args.offset))
    build_view(scene, view, tuple(args.offset))
    matched = build_matched_camera(scene, view, tuple(args.offset), args)
    scene["ekagrata_scene"] = "raw camera shadow (M5-preview), NOT the Digital Twin"
    scene["ekagrata_axis_caveat"] = core.AXIS_CAVEAT
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"[EKAGRATA] raw camera shadow scene saved: {out} ({len(bpy.data.objects)} objects)")
    print(f"[EKAGRATA] scene camera: cam_matched ({'matched view' if matched else 'alternative viewpoint'})")
    print(f"[EKAGRATA] {core.AXIS_CAVEAT}")


main()
