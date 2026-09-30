# Blender raw camera shadow (M5-preview)

The **raw camera shadow** is a stick figure in Blender whose points are the raw MediaPipe landmark positions,
streamed over UDP. It is **NOT the Digital Twin**: no fixed segment lengths, no joint rotations, no kinematic
model. (The Digital Twin is M5, built on M4's `rig.json`.)

> **Axis mapping unverified until check_axes passes.** Points are mapped to the EKAGRATA world with
> `mp_to_world` from `configs/camera.yaml`, which has not yet been confirmed by `scripts/check_axes.py`.
> The figure may be mirrored or rotated until it is. This milestone does not change the mapping.

Blender version: **5.0.1** at `D:/Video Editing/Blender/new b/blender.exe` (bundled Python 3.11.13), set in
`configs/blender.yaml`. The add-on and scene scripts use only `bpy` and the Python standard library.

All commands are PowerShell, run from the repository root (`D:\dev\ekagrata`).

## 1. Build the scene

```powershell
uv run python scripts/build_blender_preview.py
```
Writes `blender/ekagrata_preview.blend`: floor, singles court lines (13.40 × 5.18 m), net (1.524 m at the
centre), lights, camera, one Empty `lm_<name>` per landmark and one cylinder `link_<a>__<b>` per stick-figure
link, parented to `shadow_root` at `placement_offset_m`. The hitting arm (`hitting_side`) is orange.
Until data arrives the figure shows a **placeholder pose** (not data).

The figure's origin is MediaPipe's hip midpoint. `placement_offset_m` Z (default 1.0 m) is a display lift to
roughly hip height, not a measurement.

## 2. Package and install the add-on (once)

```powershell
uv run python scripts/package_blender_addon.py
```
Writes `blender/ekagrata_live.zip`. The add-on uses the **legacy `bl_info` format**: Blender 5.0.1's
*Install from Disk* still installs such zips (its source detects a zip without `blender_manifest.toml` whose
`.py` files contain `bl_info` as a legacy add-on and installs it with `preferences.addon_install`). This is
checked by `tests/test_blender_smoke.py::test_packaged_addon_installs_legacy`.

In Blender: **Edit → Preferences → Add-ons → ⌄ (top-right menu) → Install from Disk…** → choose
`D:\dev\ekagrata\blender\ekagrata_live.zip` → tick **EKAGRATA live (raw camera shadow)**.
After changing the add-on code, re-package and install again (it overwrites the old copy).

## 3. Open the scene and start the receiver

```powershell
& "D:/Video Editing/Blender/new b/blender.exe" blender/ekagrata_preview.blend
```
In the 3D viewport press **N** → tab **EKAGRATA** → check **Port** (9870, must match `udp_port`) → **Start**.
The receiver binds **127.0.0.1 only**. **Stop** closes the socket; loading another file also stops it.
On Stop, a summary is printed to the Blender system console (**Window → Toggle System Console**).

Panel values (all measured by the add-on):
- **Receive rate** / applied rate: messages per second over the last 2 s. Each ~60 Hz tick applies only the
  newest message; older ones in the same tick count as *superseded*.
- **Age since send**: `perf_counter_ns()` in Blender − `t_send_ns`, median and p95 over the last 2 s.
- **Age since capture** (live only): `perf_counter_ns()` − `t_sync_ns`. In live mode `t_sync_ns` is the host
  clock at frame grab, so this is capture → Blender apply latency (it excludes display). In replay it is
  **not meaningful** (`t_sync_ns` comes from the recording) and the panel says so.
- Both ages compare clocks of two processes. That is valid because on Windows `perf_counter_ns()` is
  QueryPerformanceCounter, one system-wide clock; it is only valid on the same machine.
- **Stale** (not newer than the last applied), **lost** (gaps in `seq`), **bad** (invalid datagrams),
  **sender restarts**.
- Landmarks with visibility < 0.5 (or missing) are hidden, together with their links.

## 4. Replay a recorded session

```powershell
uv run python scripts/stream_to_blender.py --session data/sessions/2026-09-29_swings01_002 --loop
```
Streams `pose/cam0_landmarks.parquet` in real time from `t_sync_ns` (Ctrl+C to stop; `--seconds N` to stop
after N s). The session folder is only read. On exit it prints a measured summary (messages sent, rate-limited,
send errors, measured send rate, frames with pose, replay lateness vs schedule).

## 5. Live camera

```powershell
uv run python scripts/stream_to_blender.py --camera 0
```
Uses the M1 capture + PoseLandmarker path with `configs/camera.yaml` (nothing is recorded). Add `--smooth` for
the online One-Euro filter (`one_euro_*` in `configs/joints.yaml`).

## Troubleshooting
- **"cannot bind 127.0.0.1:9870"**: another Blender (or program) uses the port. Stop it, or change
  `udp_port` in `configs/blender.yaml` and the panel's Port to the same value.
- **Nothing moves**: check the panel's *Receive rate*. 0 Hz means no packets: is `stream_to_blender.py`
  running, and are the ports equal? Received but figure hidden: landmarks have visibility < 0.5 (framing,
  lighting) — the summary's *Frames with pose* tells you whether MediaPipe saw a person.
- **Figure mirrored/tilted**: expected until `scripts/check_axes.py` passes (axis mapping unverified).
- Loopback UDP normally needs no firewall rule; if a security tool blocks it, allow `blender.exe` and the
  project's Python for 127.0.0.1.

## Headless smoke test
`uv run pytest -q tests/test_blender_smoke.py` runs the real `blender_exe`: version check, scene build +
object checks + add-on apply/keep-newest checks (`blender/tests/smoke_check.py`), and a legacy install of the
packaged zip into a temporary `BLENDER_USER_RESOURCES` folder (your real Blender config is not touched).
