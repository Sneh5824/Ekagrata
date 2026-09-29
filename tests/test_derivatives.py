import numpy as np
import pytest

from ekagrata.analysis.derivatives import resample_uniform, savgol_derivative, unwrap_nan


def test_resample_known_answer():
    t = np.array([0, 10, 30, 40], dtype=np.int64) * 1_000_000  # ms -> ns
    x = np.array([0.0, 1.0, 3.0, 4.0])
    tg, xg = resample_uniform(t, x, fs_hz=200.0, max_gap_s=1.0)  # 5 ms grid
    np.testing.assert_array_equal(tg, np.arange(0, 45, 5) * 1_000_000)
    np.testing.assert_allclose(xg, np.arange(0, 45, 5) / 10.0)


def test_resample_nan_across_long_gap_and_missing_samples():
    t = np.array([0, 10, 50, 60], dtype=np.int64) * 1_000_000
    x = np.array([[0.0], [1.0], [5.0], [np.nan]])
    tg, xg = resample_uniform(t, x, fs_hz=100.0, max_gap_s=0.015)
    assert xg.shape == (7, 1)
    np.testing.assert_allclose(xg[:2, 0], [0.0, 1.0])
    assert np.all(np.isnan(xg[2:5, 0]))  # 10..50 ms gap (40 ms) > 15 ms -> NaN
    assert xg[5, 0] == 5.0  # exactly on a sample
    assert np.isnan(xg[6, 0])  # sample itself NaN


def test_resample_rejects_non_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_uniform([0, 5, 5], [1, 2, 3], 100.0, 1.0)


def test_savgol_derivative_of_sinusoid():
    fs, f, amp = 100.0, 2.0, 0.7
    t = np.arange(0, 3, 1 / fs)
    x = amp * np.sin(2 * np.pi * f * t)
    w = 2 * np.pi * f
    d1 = savgol_derivative(x, fs, window=11, polyorder=3, deriv=1)
    d2 = savgol_derivative(x, fs, window=11, polyorder=3, deriv=2)
    core = slice(10, -10)
    # Tolerances relative to the true peak: 1% for d1; 5% for d2, because SG smoothing (window 11 at 100 Hz)
    # attenuates the 2nd derivative of a 2 Hz sine (observed error in this test: 3.2% of peak).
    assert np.max(np.abs(d1[core] - amp * w * np.cos(w * t[core]))) < 0.01 * amp * w
    assert np.max(np.abs(d2[core] + amp * w**2 * np.sin(w * t[core]))) < 0.05 * amp * w**2


def test_savgol_derivative_not_across_gaps():
    fs = 100.0
    x = np.arange(40, dtype=float) / fs * 2.0  # slope 2 per s
    x[15] = np.nan
    x[30:35] = np.nan
    d = savgol_derivative(x, fs, window=7, polyorder=2, deriv=1)
    np.testing.assert_allclose(d[:15], 2.0, atol=1e-9)
    np.testing.assert_allclose(d[16:30], 2.0, atol=1e-9)
    assert np.isnan(d[15]) and np.all(np.isnan(d[30:35]))
    assert np.all(np.isnan(d[35:]))  # run of 5 < window 7


def test_savgol_bad_params():
    with pytest.raises(ValueError):
        savgol_derivative(np.zeros(20), 10.0, window=6, polyorder=2)


def test_unwrap_nan():
    a = np.array([3.0, -3.0, np.nan, 3.1, -3.1])
    out = unwrap_nan(a)
    assert out[1] == pytest.approx(-3.0 + 2 * np.pi)
    assert np.isnan(out[2])
    assert out[4] == pytest.approx(-3.1 + 2 * np.pi)
