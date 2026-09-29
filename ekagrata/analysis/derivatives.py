"""Uniform resampling and NaN-aware Savitzky-Golay derivatives on the t_sync_ns timeline."""

import numpy as np
from scipy.signal import savgol_filter

from ekagrata.analysis.filters import finite_runs


def resample_uniform(t_ns, x, fs_hz: float, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling of x (N, ...) at strictly increasing times t_ns onto a uniform grid at fs_hz,
    starting at t_ns[0]. A grid point is NaN unless both bracketing samples are finite and at most
    max_gap_s apart. Returns (t_grid_ns int64, x_grid)."""
    t = np.asarray(t_ns, dtype=np.int64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_ns must be 1-D with at least 2 samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_ns must be strictly increasing")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    x = np.asarray(x, dtype=np.float64)
    cols = x.reshape(x.shape[0], -1)
    step_ns = 1e9 / fs_hz
    n_grid = int(np.floor((t[-1] - t[0]) / step_ns)) + 1
    t_grid = t[0] + np.round(np.arange(n_grid) * step_ns).astype(np.int64)
    j = np.clip(np.searchsorted(t, t_grid, side="right") - 1, 0, t.size - 2)
    t0, t1 = t[j], t[j + 1]
    w = ((t_grid - t0) / (t1 - t0).astype(np.float64))[:, None]
    x0, x1 = cols[j], cols[j + 1]
    out = (1.0 - w) * x0 + w * x1
    exact = (w[:, 0] == 0.0)[:, None]
    out = np.where(exact, x0, out)  # grid point on a sample: needs only that sample
    gap_ok = ((t1 - t0) <= max_gap_s * 1e9)[:, None]
    bracket_ok = np.isfinite(x0) & np.isfinite(x1) & gap_ok
    out = np.where(exact | bracket_ok, out, np.nan)
    return t_grid, out.reshape((n_grid,) + x.shape[1:])


def savgol_derivative(x, fs_hz: float, window: int, polyorder: int, deriv: int = 1) -> np.ndarray:
    """deriv-th time derivative (units of x per s^deriv) of x (N, ...) sampled uniformly at fs_hz.
    Computed per finite run; runs shorter than `window` give NaN, so nothing is differentiated across a
    gap."""
    if window % 2 != 1 or window <= polyorder or deriv > polyorder:
        raise ValueError("need odd window > polyorder >= deriv")
    x = np.asarray(x, dtype=np.float64)
    cols = x.reshape(x.shape[0], -1)
    out = np.full_like(cols, np.nan)
    for k in range(cols.shape[1]):
        c = cols[:, k]
        for start, stop in finite_runs(np.isfinite(c)):
            if stop - start >= window:
                out[start:stop, k] = savgol_filter(c[start:stop], window, polyorder, deriv=deriv,
                                                   delta=1.0 / fs_hz, mode="interp")
    return out.reshape(x.shape)


def unwrap_nan(angle_rad) -> np.ndarray:
    """np.unwrap applied to each finite run of a 1-D angle series (NaNs kept)."""
    a = np.asarray(angle_rad, dtype=np.float64)
    out = a.copy()
    for start, stop in finite_runs(np.isfinite(a)):
        out[start:stop] = np.unwrap(a[start:stop])
    return out
