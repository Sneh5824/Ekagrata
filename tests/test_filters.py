import numpy as np
import pytest

from ekagrata.analysis.filters import (
    OneEuroFilter,
    butter_filtfilt,
    finite_runs,
    interpolate_gaps,
    visibility_gate,
)


def test_finite_runs():
    assert finite_runs([True, True, False, True]) == [(0, 2), (3, 4)]
    assert finite_runs([False, False]) == []


def test_visibility_gate():
    x = np.ones((2, 3, 3))
    vis = np.array([[0.9, 0.2, np.nan], [0.5, 0.6, 0.49]])
    out = visibility_gate(x, vis, 0.5)
    assert np.isfinite(out[0, 0]).all() and np.isnan(out[0, 1]).all() and np.isnan(out[0, 2]).all()
    assert np.isfinite(out[1, 0]).all() and np.isnan(out[1, 2]).all()
    assert np.all(x == 1.0)  # input untouched


def test_interpolate_gaps_fills_only_short_interior_gaps():
    x = np.array([np.nan, 0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    out = interpolate_gaps(x, max_gap_frames=2)
    np.testing.assert_allclose(out[1:5], [0, 1, 2, 3])
    assert np.isnan(out[0]) and np.isnan(out[-1])  # leading / trailing kept
    assert np.all(np.isnan(out[5:9]))  # gap of 4 > 2 kept
    assert out[9] == 8.0


def test_interpolate_gaps_multicolumn_shape():
    x = np.zeros((5, 2, 3))
    x[2, 1, 0] = np.nan
    out = interpolate_gaps(x, 1)
    assert out.shape == x.shape and np.all(np.isfinite(out))


def test_filtfilt_zero_phase_lag_and_passband():
    fs, f = 100.0, 1.0
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * f * t)
    y = butter_filtfilt(x, fs, cutoff_hz=5.0, order=4)
    lags = np.arange(-20, 21)
    core = slice(100, -100)
    xc = [np.dot(x[core], np.roll(y, -k)[core]) for k in lags]
    assert lags[int(np.argmax(xc))] == 0  # phase lag ~ 0 samples
    np.testing.assert_allclose(y[core], x[core], atol=0.01)


def test_filtfilt_attenuates_high_frequency():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    y = butter_filtfilt(np.sin(2 * np.pi * 30 * t), fs, cutoff_hz=5.0)
    assert np.max(np.abs(y[50:-50])) < 0.01


def test_filtfilt_nan_segments_not_bridged():
    fs = 50.0
    x = np.ones(100)
    x[40:45] = np.nan
    x[90:] = 5.0
    x[88:90] = np.nan
    y = butter_filtfilt(x, fs, cutoff_hz=5.0, order=4)
    assert np.all(np.isnan(y[40:45]))
    np.testing.assert_allclose(y[:40], 1.0, atol=1e-9)  # constant runs are unchanged
    np.testing.assert_allclose(y[45:88], 1.0, atol=1e-9)
    assert np.all(np.isnan(y[88:]))  # tail run of 10 samples < 3 * order = 12 -> NaN


def test_filtfilt_cutoff_above_nyquist_rejected():
    with pytest.raises(ValueError, match="fs/2"):
        butter_filtfilt(np.zeros(100), fs_hz=15.0, cutoff_hz=7.5)


def test_one_euro_converges_on_constant_and_follows_step():
    f = OneEuroFilter(min_cutoff_hz=1.0, beta=0.0, d_cutoff_hz=1.0)
    t = 0
    for _ in range(5):
        out = f(t, np.array([2.0, -1.0]))
        t += 10_000_000
    np.testing.assert_allclose(out, [2.0, -1.0])
    for _ in range(300):  # 3 s after a step: converged
        out = f(t, np.array([3.0, -1.0]))
        t += 10_000_000
    np.testing.assert_allclose(out, [3.0, -1.0], atol=1e-6)


def test_one_euro_lags_less_with_beta():
    def run(beta):
        f = OneEuroFilter(min_cutoff_hz=0.5, beta=beta, d_cutoff_hz=1.0)
        outs = [f(i * 10_000_000, np.array([float(i)]))[0] for i in range(100)]  # ramp 100 units/s
        return 99.0 - outs[-1]

    assert run(1.0) < run(0.0)


def test_one_euro_nan_resets_element():
    f = OneEuroFilter()
    f(0, np.array([1.0]))
    assert np.isnan(f(10_000_000, np.array([np.nan]))[0])
    assert f(20_000_000, np.array([5.0]))[0] == 5.0
