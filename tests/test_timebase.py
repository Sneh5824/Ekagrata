import numpy as np
import pytest

from ekagrata.core import timebase

U32 = 2**32


def test_now_ns_is_monotonic_int():
    a = timebase.now_ns()
    b = timebase.now_ns()
    assert isinstance(a, int)
    assert b >= a


def test_unwrap_no_wrap_is_unchanged():
    v = [0, 10, 20, 30]
    out = timebase.unwrap_u32(v)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, v)


def test_unwrap_single_wrap():
    v = [U32 - 20, U32 - 10, 5, 15]
    out = timebase.unwrap_u32(v)
    np.testing.assert_array_equal(out, [U32 - 20, U32 - 10, U32 + 5, U32 + 15])
    assert np.all(np.diff(out) > 0)


def test_unwrap_two_wraps():
    v = [U32 - 100, 50, U32 - 50, 100]
    out = timebase.unwrap_u32(v)
    np.testing.assert_array_equal(out, [U32 - 100, U32 + 50, 2 * U32 - 50, 2 * U32 + 100])
    assert np.all(np.diff(out) > 0)


def test_unwrap_empty_and_single():
    assert timebase.unwrap_u32([]).size == 0
    np.testing.assert_array_equal(timebase.unwrap_u32([7]), [7])


def test_unwrap_rejects_out_of_range():
    with pytest.raises(ValueError):
        timebase.unwrap_u32([0, U32])
    with pytest.raises(ValueError):
        timebase.unwrap_u32([-1, 0])


def test_ns_to_s():
    np.testing.assert_allclose(timebase.ns_to_s([0, 1_500_000_000, 2_000_000_000]), [0.0, 1.5, 2.0])
