"""Tests for wienf -- Wiener filter."""
import numpy as np
from morie.fn.wienf import wienf
from morie.fn._containers import SignalResult


def test_wienf_basic(signal_1khz):
    x, fs = signal_1khz
    result = wienf(x, fs)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert len(result.filtered) == len(x)


def test_wienf_reduces_noise():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    signal = np.sin(2 * np.pi * 100 * t)
    noisy = signal + rng.standard_normal(len(t)) * 2.0
    result = wienf(noisy, fs)
    noise_before = np.std(noisy - signal)
    noise_after = np.std(result.filtered - signal)
    assert noise_after < noise_before


def test_wienf_gain_range():
    x = np.random.default_rng(7).standard_normal(256)
    result = wienf(x)
    gain = result.extra["wiener_gain"]
    assert np.all(gain >= 0)
    assert np.all(gain <= 1.0 + 1e-6)
