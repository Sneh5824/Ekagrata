"""Time helpers. All timestamps are int64 nanoseconds (`_ns`) or ESP32 microseconds (`_us`)."""

import time

import numpy as np

_U32_SPAN = np.int64(2**32)


def now_ns() -> int:
    """Host monotonic clock in nanoseconds (never time.time())."""
    return time.perf_counter_ns()


def unwrap_u32(values) -> np.ndarray:
    """Unwrap a sequence of uint32 counters (e.g. ESP32 micros(), wraps at 2^32) into monotonic int64.

    A wrap is detected wherever a sample is smaller than its predecessor.
    """
    v = np.asarray(values, dtype=np.int64)
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {v.shape}")
    if v.size and (v.min() < 0 or v.max() >= _U32_SPAN):
        raise ValueError("values must be in the uint32 range [0, 2^32)")
    if v.size < 2:
        return v.copy()
    wraps = np.concatenate([[0], np.cumsum(np.diff(v) < 0)]).astype(np.int64)
    return v + wraps * _U32_SPAN


def ns_to_s(t_ns) -> np.ndarray:
    """Convert integer nanoseconds to float seconds."""
    return np.asarray(t_ns, dtype=np.int64).astype(np.float64) * 1e-9
