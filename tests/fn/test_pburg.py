"""Tests for pburg — Burg AR power spectral density."""
import numpy as np
from moirais.fn.pburg import burg_psd
from moirais.fn._containers import SignalResult


def test_pburg_basic(signal_1khz):
    x, fs = signal_1khz
    result = burg_psd(x, fs, order=16)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert "freqs" in result.extra


def test_pburg_peak_at_signal_freq(rng):
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t) + rng.standard_normal(len(t)) * 0.1
    result = burg_psd(x, fs, order=32, nfft=512)
    freqs = result.extra["freqs"]
    psd = result.filtered
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 100) < 10
