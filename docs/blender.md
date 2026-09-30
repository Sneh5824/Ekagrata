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
uv run python scripts/stream_to_blender.py
```
The camera is `device` in `configs/camera.yaml` (`--camera N` overrides it). On 2026-09-30 DirectShow index 0
was the Iriun virtual camera (phone app running, showing its "waiting" screen) and index 1 the laptop webcam,
so `device: 1`. The startup `Camera:` line prints the index and backend; the capture refuses a camera whose
reported fourcc differs from the requested one or whose first frames are near-black. Uses the M1 capture + PoseLandmarker path with `configs/camera.yaml` (nothing is recorded). Add `--smooth` for
the online One-Euro filter (`one_euro_*` in `configs/joints.yaml`).

## Latency breakdown method (basis for experiment E07)

Goal: split "age since capture" (Blender apply time − `t_sync_ns`) into stages, all measured with the
host clock `time.perf_counter_ns()` (QueryPerformanceCounter, system-wide on Windows, so stamps from the
streamer and from Blender are comparable on the same machine).

**Stamps** (live mode, one row per SENT message; `--timing-csv PATH` writes them per frame):

| stamp | where | meaning |
|---|---|---|
| `t_grab_ns` | capture thread, right after `cap.grab()` | diagnostic only (DirectShow: before the frame arrives) |
| `t_host_ns` | capture thread, right after `cap.retrieve()` | = `t_sync_ns`; frame arrival + decode ("after_retrieve") |
| `t_dequeue_ns` | streamer, when the frame leaves the capture queue | |
| `t_infer_start_ns` / `t_infer_end_ns` | around `detect()` | includes BGR→RGB and `mp.Image` |
| `t_send_ns` | sender, before JSON encoding | also sent in the message |
| `t_sent_ns` | sender, after `sendto()` returns | |
| apply time | Blender add-on, when the message is applied | gives "age since send" and "age since capture" |

**Stages** (printed by `stream_to_blender.py` on exit as median / p95 / max): grab → retrieve end =
t_host − grab (DirectShow: mostly waiting for the frame, not part of capture age); queue wait = dequeue −
t_host; dequeue → inference; inference; map + smooth = send − infer_end; encode + sendto = sent − send;
t_host → sent. Blender adds send → apply (panel / console "age since send"). Queue wait is also reported in
units of the median frame interval.

**Consistency check:** median(t_host → sent) + median(Blender age since send) should be close to the
Blender median "age since capture" (medians do not add exactly; a large mismatch means a missing stage).

**How to run** (Blender open, add-on started, person in frame):
```powershell
uv run python scripts/stream_to_blender.py --seconds 30 --timing-csv experiments/E07_latency/raw/<name>.csv
```
then press **Stop** in Blender and copy the console summary.

**Known measurement caveats:**
- With OpenCV's DirectShow backend, `grab()` returns almost immediately and `retrieve()` blocks until the next
  frame arrives. Since 2026-09-30 `t_host_ns` is therefore stamped after `retrieve()` ("after_retrieve");
  sessions and timing files from before that were stamped after `grab()`, about one frame interval early, and
  their "capture age" includes that wait (see `experiments/E07_latency/README.md`).
- "after_retrieve" is frame arrival + decode (≈ 3 ms), not exposure time.
- Exposure, sensor readout, USB transfer and driver time before the frame reaches OpenCV are not visible to the
  host clock; measuring them needs an external reference (e.g. the M7 sync LED or a high-speed video of a
  screen and an event).
- Display is excluded: Blender viewport redraw and monitor latency come after "apply".
- The add-on polls at ~60 Hz, so send → apply includes up to one timer tick (~16.7 ms) of waiting.
- Camera index: DirectShow and MSMF number devices differently, and a virtual camera (Iriun) can take index 0.
  Confirm the device with the printed `Camera:` line (requested vs driver-reported fourcc) and `Frames with pose`.

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
