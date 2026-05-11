"""Tests for buttbp — Butterworth bandpass filter."""
import numpy as np
from morie.fn.buttbp import butter_bandpass
from morie.fn._containers import SignalResult


def test_buttbp_basic(signal_1khz):
    x, fs = signal_1khz
    result = butter_bandpass(x, fs, low=800, high=1200)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_buttbp_isolates_band(signal_1khz):
    x, fs = signal_1khz
    result = butter_bandpass(x, fs, low=800, high=1200)
    fft_filt = np.abs(np.fft.rfft(result.filtered))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_1k = np.argmin(np.abs(freqs - 1000))
    assert fft_filt[idx_1k] > fft_filt[idx_50] * 10
