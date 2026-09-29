"""NaN-aware filters. Arrays are time-major: shape (N, ...), filtered independently per trailing element.

Offline: visibility gating, short-gap interpolation, zero-phase Butterworth (sosfiltfilt).
Online: One-Euro filter (Casiez, Roussel & Vogel, CHI 2012).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _columns(x) -> tuple[np.ndarray, tuple]:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1), x.shape


def finite_runs(mask) -> list[tuple[int, int]]:
    """[start, stop) index pairs of consecutive True values in a 1-D boolean array."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def visibility_gate(x, vis, threshold: float) -> np.ndarray:
    """Set x to NaN where visibility < threshold (or is NaN). x: (N, K, ...) and vis: (N, K)."""
    x = np.array(x, dtype=np.float64, copy=True)
    vis = np.asarray(vis, dtype=np.float64)
    bad = ~(vis >= threshold)  # NaN visibility counts as bad
    x[bad] = np.nan
    return x


def interpolate_gaps(x, max_gap_frames: int) -> np.ndarray:
    """Linearly fill interior NaN runs of length <= max_gap_frames (uniform sampling assumed).
    Longer runs and leading/trailing NaNs are left as NaN."""
    cols, shape = _columns(x)
    out = cols.copy()
    idx = np.arange(cols.shape[0])
    for j in range(cols.shape[1]):
        c = cols[:, j]
        finite = np.isfinite(c)
        for start, stop in finite_runs(~finite):
            if start == 0 or stop == len(c) or stop - start > max_gap_frames:
                continue
            out[start:stop, j] = np.interp(idx[start:stop], [start - 1, stop], [c[start - 1], c[stop]])
    return out.reshape(shape)


def butter_filtfilt(x, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass applied to each finite run separately (never across NaN gaps).
    Runs shorter than 3 * order samples cannot be filtered reliably and become NaN."""
    if not 0 < cutoff_hz < fs_hz / 2:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be in (0, fs/2) = (0, {fs_hz / 2:.3f}) Hz")
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    default_pad = 3 * (2 * len(sos) + 1)
    cols, shape = _columns(x)
    out = np.full_like(cols, np.nan)
    for j in range(cols.shape[1]):
        c = cols[:, j]
        for start, stop in finite_runs(np.isfinite(c)):
            n = stop - start
            if n < 3 * order:
                continue
            out[start:stop, j] = sosfiltfilt(sos, c[start:stop], padlen=min(default_pad, n - 1))
    return out.reshape(shape)


class OneEuroFilter:
    """Online One-Euro filter for arrays of any shape. Speed-adaptive cutoff:
    cutoff = min_cutoff_hz + beta * |dx/dt|. A NaN input returns NaN and resets that element."""

    def __init__(self, min_cutoff_hz: float = 1.0, beta: float = 0.0, d_cutoff_hz: float = 1.0):
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.d_cutoff_hz = d_cutoff_hz
        self._x = None
        self._dx = None
        self._t_ns: int | None = None

    @staticmethod
    def _alpha(cutoff_hz, dt_s):
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def __call__(self, t_ns: int, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t_ns is None or t_ns <= self._t_ns:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t_ns = t_ns
            return x.copy()
        dt_s = (t_ns - self._t_ns) * 1e-9
        self._t_ns = t_ns
        fresh = ~np.isfinite(self._x)  # elements reset after a NaN start over from the new value
        dx = np.where(fresh, 0.0, (x - self._x) / dt_s)
        a_d = self._alpha(self.d_cutoff_hz, dt_s)
        self._dx = np.where(fresh, 0.0, a_d * dx + (1.0 - a_d) * self._dx)
        cutoff = self.min_cutoff_hz + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt_s)
        self._x = np.where(fresh, x, a * x + (1.0 - a) * self._x)
        self._dx = np.where(np.isfinite(self._x), self._dx, 0.0)
        return self._x.copy()
