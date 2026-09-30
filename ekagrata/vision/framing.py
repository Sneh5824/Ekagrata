"""Camera framing checks for recording and live previews (M2 framing fixes).

MediaPipe outputs image coordinates even for landmarks outside the image, so "in frame" (0 <= x, y <= 1) is
checked separately from visibility. Image y grows downward: y < 0 is above the top edge, y > 1 below the
bottom edge.
"""

import cv2
import numpy as np

from ekagrata.vision.landmark_map import LM

VIS_THRESHOLD = 0.5
# The racket is not a landmark and extends ~0.6 m above the wrist: a wrist this close to the top edge
# (normalised image height) means the racket is already cut off.
NEAR_TOP_Y = 0.10

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)


def side_prefix(side: str) -> str:
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return "r_" if side == "right" else "l_"


def framing_landmarks(side: str = "right") -> tuple[str, ...]:
    """Landmarks shown in the framing indicator: both hips, hitting-side elbow and wrist."""
    s = side_prefix(side)
    return ("l_hip", "r_hip", f"{s}elbow", f"{s}wrist")


def framing_status(landmarks_img, side: str = "right", threshold: float = VIS_THRESHOLD) -> dict[str, bool]:
    """{name: ok} for framing_landmarks(side). ok = visibility >= threshold AND inside the image.
    landmarks_img is MediaPipe's (33, 5) array (x, y, z, visibility, presence); NaN (no pose) is not ok."""
    lm = np.asarray(landmarks_img, dtype=np.float64)
    out = {}
    for name in framing_landmarks(side):
        x, y, _, vis, _ = lm[LM[name]]
        out[name] = bool(vis >= threshold and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return out


def wrist_edge(landmarks_img, side: str = "right", near_top_y: float = NEAR_TOP_Y) -> str | None:
    """Hitting-side wrist relative to the image edges: "above_top" (y < 0), "below_bottom" (y > 1),
    "near_top" (0 <= y < near_top_y), or None (inside and not near the top, or no pose)."""
    y = float(np.asarray(landmarks_img, dtype=np.float64)[LM[f"{side_prefix(side)}wrist"], 1])
    if not np.isfinite(y):
        return None
    if y < 0.0:
        return "above_top"
    if y > 1.0:
        return "below_bottom"
    if y < near_top_y:
        return "near_top"
    return None


def wrist_edge_message(edge: str | None, side: str = "right") -> tuple[str, tuple] | None:
    """(text, BGR colour) for a wrist_edge result: red when out of frame, yellow when near the top."""
    arm = side.upper()
    return {
        "above_top": (f"{arm} WRIST ABOVE FRAME TOP", RED),
        "below_bottom": (f"{arm} WRIST BELOW FRAME BOTTOM", RED),
        "near_top": (f"{arm} WRIST NEAR FRAME TOP - racket cut off", YELLOW),
    }.get(edge)


def rest_gate(vis_samples: dict, threshold: float = VIS_THRESHOLD) -> list[tuple[str, float]]:
    """Landmarks whose MEDIAN visibility over the rest window is below threshold, as (name, median).
    Samples are per-frame visibilities; NaN (frame without a pose) counts as 0. No samples fails (0)."""
    failed = []
    for name, values in vis_samples.items():
        v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
        med = float(np.median(v)) if v.size else 0.0
        if med < threshold:
            failed.append((name, med))
    return failed


def draw_framing(image: np.ndarray, status: dict[str, bool], edge: str | None, side: str = "right") -> None:
    """Draw the framing indicator in place on a BGR preview image: one green/red label per landmark
    (top-right), and a red (out of frame) or yellow (near top) wrist banner along the bottom."""
    h, w = image.shape[:2]
    x0 = max(w - 190, 0)
    for k, (name, ok) in enumerate(status.items()):
        y = 28 + 26 * k
        colour = GREEN if ok else RED
        cv2.rectangle(image, (x0, y - 16), (x0 + 16, y), colour, -1)
        cv2.putText(image, f"{name} {'OK' if ok else 'NOT IN VIEW'}", (x0 + 24, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    msg = wrist_edge_message(edge, side)
    if msg is not None:
        text, colour = msg
        cv2.rectangle(image, (0, h - 44), (w, h), colour, -1)
        cv2.putText(image, text, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
