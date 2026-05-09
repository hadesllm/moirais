"""Tests for wnflt -- Wiener filter."""
import numpy as np
from moirais.fn.wnflt import wiener_filter
from moirais.fn._containers import SignalResult


def test_wiener_basic(signal_1khz):
    x, fs = signal_1khz
    result = wiener_filter(x, fs)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert len(result.filtered) == len(x)
    assert "wiener_gain" in result.extra
    assert "noise_psd" in result.extra


def test_wiener_reduces_noise():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    clean = np.sin(2 * np.pi * 50 * t)
    noisy = clean + rng.standard_normal(len(t)) * 2.0
    result = wiener_filter(noisy, fs)
    noise_before = np.std(noisy - clean)
    noise_after = np.std(result.filtered - clean)
    assert noise_after < noise_before


def test_wiener_with_known_noise_psd():
    rng = np.random.default_rng(7)
    fs = 500
    t = np.arange(0, 0.5, 1 / fs)
    x = np.sin(2 * np.pi * 20 * t) + rng.standard_normal(len(t)) * 0.5
    noise_psd = np.ones(len(t) // 2 + 1) * 0.25
    result = wiener_filter(x, fs, noise_psd=noise_psd)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)
