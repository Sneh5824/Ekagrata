# E07 latency — raw evidence (protocol to be written in M14)

Per-frame latency stamps from `scripts/stream_to_blender.py --timing-csv` (method: `docs/blender.md`,
"Latency breakdown method"). All stamps are host `time.perf_counter_ns()` (QueryPerformanceCounter), one row per
UDP message sent to the Blender raw camera shadow. These are REAL measurements, not simulated.

## raw/2026-09-30_breakdown_seated_pre_fix.csv

- **What:** 30 s live run, 897 sent messages, laptop webcam, Blender 5.0.1 GUI receiving (add-on started).
  The user was seated at the laptop, in frame (pose detected in 100 % of frames). Not standing.
- **When:** 2026-09-30, before the timestamp-convention fix.
- **Settings:** `configs/camera.yaml` at the time with `--camera 1` (DirectShow index 1 = USB2.0 HD UVC WebCam),
  dshow, 1280x720, MJPG, `exposure_auto: false`, `exposure: -6.0`; driver reported 1280x720, fourcc MJPG,
  exposure -6.0. PoseLandmarker `full`, CPU. Smoothing off. Sender rate limit 60 Hz (0 messages limited).
  Measured: capture 29.89 fps, 0 frames dropped, send rate 29.95 Hz.
- **Timestamp convention: PRE-FIX.** Column meanings in this file:
  - `t_grab_ns`: after `cap.grab()`. At the time this WAS `t_host_ns` = `t_sync_ns` ("after_grab"). With OpenCV
    DirectShow, `grab()` returns before the frame arrives, so this stamp is ≈ one frame interval early.
  - `t_retrieved_ns`: after `cap.retrieve()` = frame arrival + decode. Equivalent to the post-fix `t_host_ns`
    ("after_retrieve").
  - `t_dequeue_ns`, `t_infer_start_ns`, `t_infer_end_ns`, `t_send_ns` (before JSON encoding), `t_sent_ns`
    (after `sendto`), plus `frame_idx`, `dropped_before`, `pose` (1 = pose detected).
- **Blender side of the same run** (not in the file; add-on console on Stop, 716 messages applied, 0 lost):
  age since send median 8.5 ms / p95 17.8 ms; age since capture median 58.4 ms / p95 78.4 ms. That capture age
  was measured from the PRE-FIX `t_sync_ns` (= `t_grab_ns`), so it includes the ≈ one-frame `retrieve()` wait.
- **Direct probe (same day, same settings, 300 frames):** `grab()` median 0.01 ms, `retrieve()` median
  32.00 ms (p95 47.80 ms); `cv2.imdecode` of a 1280x720 JPEG median 2.85 ms.

## raw/2026-09-30_breakdown_seated_post_fix.csv

- **What:** same setup as the pre-fix file, 30 s, 900 sent messages, user seated in frame (pose 100 %), Blender
  5.0.1 GUI receiving. Measured: capture 30.00 fps, 0 dropped, send rate 30.01 Hz.
- **When:** 2026-09-30, after the timestamp-convention fix.
- **Timestamp convention: POST-FIX.** `t_host_ns` = `t_sync_ns` = after `cap.retrieve()` ("after_retrieve":
  frame arrival + decode, not exposure time); `t_grab_ns` = after `cap.grab()` (diagnostic only). Other
  columns as above.
- **Blender side** (739 applied, 0 lost): age since send median 8.2 ms / p95 18.0 ms; age since capture
  median 26.5 ms / p95 36.2 ms (capture = after_retrieve, so the camera-internal latency before arrival is
  NOT included; display is not included).
