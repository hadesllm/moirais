"""Tests for buttlp — Butterworth lowpass filter."""
import numpy as np
from morie.fn.buttlp import butter_lowpass
from morie.fn._containers import SignalResult


def test_buttlp_basic(signal_1khz):
    x, fs = signal_1khz
    result = butter_lowpass(x, fs, cutoff=200)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_buttlp_removes_high_freq(signal_1khz):
    x, fs = signal_1khz
    result = butter_lowpass(x, fs, cutoff=200)
    fft_orig = np.abs(np.fft.rfft(x))
    fft_filt = np.abs(np.fft.rfft(result.filtered))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    idx_1k = np.argmin(np.abs(freqs - 1000))
    assert fft_filt[idx_1k] < fft_orig[idx_1k] * 0.01
