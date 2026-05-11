"""Tests for chflt -- Chebyshev type I filter."""
import numpy as np
from morie.fn.chflt import chflt
from morie.fn._containers import SignalResult


def test_chflt_basic(signal_1khz):
    x, fs = signal_1khz
    result = chflt(x, fs, cutoff=200.0, btype="low")
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_chflt_lowpass_attenuates(signal_1khz):
    x, fs = signal_1khz
    result = chflt(x, fs, cutoff=200.0, btype="low")
    fft_orig = np.abs(np.fft.rfft(x))
    fft_filt = np.abs(np.fft.rfft(result.filtered))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    idx_1k = np.argmin(np.abs(freqs - 1000))
    assert fft_filt[idx_1k] < fft_orig[idx_1k] * 0.1


def test_chflt_highpass():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)
    result = chflt(x, fs, cutoff=100.0, btype="high")
    assert len(result.filtered) == len(x)


def test_chflt_preserves_length():
    x = np.random.default_rng(7).standard_normal(512)
    result = chflt(x, 500.0, cutoff=50.0)
    assert len(result.filtered) == 512
