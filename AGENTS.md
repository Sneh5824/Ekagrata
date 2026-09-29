# EKAGRATA — Agent Rules (read before every task)

EKAGRATA = athlete Digital Twin + Motion Shadow for the **badminton forehand smash**.
Pipeline: camera + 3 wearable IMUs (BNO055 + ESP32, ESP-NOW → USB receiver) → sync → fusion →
joint angles / racket state → Digital Twin (Blender, RViz/MuJoCo) → smash simulation → ML → optimization → (RL, later).
Robofest Gujarat 6.0 Grand Finale project. Judges will ask for measured evidence, so correctness beats speed.

## 1. Platform
- Windows 11, PowerShell. Give PowerShell commands, not bash.
- Python **3.12** (MediaPipe supports 3.9–3.12). Environment and deps managed with **uv** (`uv sync`, `uv run ...`).
  Never `pip install` into the global Python. Never create a second venv.
- Laptop GPU: RTX 4050. Do not assume MediaPipe uses the GPU on Windows.
- Every text-file open/read/write (`open()`, `Path.read_text`/`write_text`, pandas `read_csv`/`to_csv`,
  `json.dump`) must pass `encoding="utf-8"` explicitly. Binary modes (`"rb"`, `"wb"`, Parquet) are exempt: they
  take no encoding.
- Never delete, move or rename a file you did not create in the current milestone without asking the user first.

## 2. Source of truth and scope discipline
- **`docs/SPEC.md` is the source of truth** for architecture, formats and milestones. `docs/PROGRESS.md` records
  what is done. `docs/reference/Ekagrata_original_proposal.pdf` is the ORIGINAL proposal: context only, it contains
  known errors listed in SPEC §1.3, and SPEC overrides it wherever they differ.
- Work on exactly ONE milestone at a time, in SPEC order, and STOP at its gate. Do not implement future milestones
  "while you're at it".
- Dependencies are added only in the milestone that SPEC says introduces them (e.g. MuJoCo in M8b, scikit-learn in
  M12, PyTorch/SB3 in M13, ROS 2 in M10). Gazebo is cut from the project.
- Before changing architecture, folder layout, conventions or adding a dependency: stop and ask, with a one-line reason.
- Prefer small, readable modules over frameworks. No classes where a function suffices.

## 3. Architecture rules
- `ekagrata/` is a pure Python package with **no ROS imports**. ROS 2 nodes (later) live in `ros2_ws/` as thin adapters.
- Data sources are swappable behind one interface: real, simulated and replayed IMU data must produce the same types.
- Blender runs in its own Python. Never import mediapipe/numpy-heavy pipeline code inside Blender; feed it over UDP.
- MediaPipe: use the **Tasks API `PoseLandmarker`** only. Do NOT use legacy `mp.solutions.pose`.
- Exactly ONE OpenCV distribution may be installed (mediapipe may pull `opencv-contrib-python`; do not also add `opencv-python`).

## 4. Conventions (non-negotiable; all enforced by tests)
- **Units:** SI internally (m, s, rad, rad/s, m/s²). Degrees only for display, and such variables/columns end in `_deg`.
- **Quaternions:** Hamilton, stored as numpy arrays **[w, x, y, z]** (scalar-first). SciPy uses [x, y, z, w]:
  convert ONLY through `ekagrata/core/quat.py`. Never call `scipy.spatial.transform.Rotation.from_quat` elsewhere.
- **Frame naming:** `q_a_b` rotates a vector expressed in frame b into frame a: `v_a = q_a_b ⊗ v_b ⊗ conj(q_a_b)`.
  Composition: `q_a_c = q_a_b ⊗ q_b_c`. Joint rotation: `q_parent_child = conj(q_world_parent) ⊗ q_world_child`.
- **World frame:** right-handed, Z up (ROS REP-103: x forward, y left, z up). MediaPipe axes are converted to this
  frame in exactly one place (`ekagrata/vision/landmark_map.py`, later).
- **Time:** integer nanoseconds, `int64`, names end in `_ns` (or `_us` for ESP32 micros). Host clock =
  `time.perf_counter_ns()`. Never use `time.time()` for measuring intervals. Canonical timeline column: `t_sync_ns`.
- **Segment / node names:** `upper_arm`, `forearm`, `racket` (or `hand` if configured), `torso`.
- **Data:** raw session data under `data/sessions/<YYYY-MM-DD>_<session>_<athlete>/` is append-only — never modify
  or delete it. Raw streams = CSV; derived tables = Parquet; metadata = `session.json`.
  Every IMU row carries `source` ∈ {`real`, `sim`, `replay`}.

## 5. Scientific honesty (critical)
- Never invent, hardcode, or "estimate" accuracy, latency, sample-rate or performance numbers in code, comments,
  docs, or chat. Every reported number must be computed from recorded data by code in this repo.
- Do not describe simulated results as real. Label simulated data and plots as simulated.
- Do not call any optimizer output "optimal" or "perfect"; say "better under the defined simulation objective".
- If something cannot be verified, say so plainly.

## 6. Definition of done (for every task)
1. `uv run ruff check .` passes.
2. `uv run pytest -q` passes, and new math/logic has tests (known-answer tests, round-trips, edge cases).
3. You ran both commands and show their actual output. Do not claim a test passed without running it.
4. Every acceptance criterion of the milestone in SPEC is checked off with evidence (command + real output).
5. `docs/PROGRESS.md` updated: milestone, date, files changed, test count, acceptance evidence, known
   limitations, open questions for the user, and the exact `git commit` message to use.
6. Then STOP and wait for the user. Never start the next milestone without the user saying so.

## 7. Repository layout (target; create folders only when a task needs them)
```
configs/  firmware/  ekagrata/{core,io,sources,vision,kinematics,fusion,sim,analysis,ml,optim,rl,viz,transport,pipeline}/
ros2_ws/src/  blender/  generated/  scripts/  experiments/  tests/  docs/  data/ (gitignored)  models/ (gitignored)
```
