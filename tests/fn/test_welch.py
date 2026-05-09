"""Tests for welch — Welch power spectral density."""
import numpy as np
from moirais.fn.welch import welch_psd
from moirais.fn._containers import SignalResult


def test_welch_basic(signal_1khz):
    x, fs = signal_1khz
    result = welch_psd(x, fs)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert "freqs" in result.extra


def test_welch_peak_at_signal_freq(signal_1khz):
    x, fs = signal_1khz
    result = welch_psd(x, fs, nperseg=1024)
    freqs = result.extra["freqs"]
    psd = result.filtered
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 1000) < 50
