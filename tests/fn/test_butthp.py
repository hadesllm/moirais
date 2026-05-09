"""Tests for butthp — Butterworth highpass filter."""
import numpy as np
from moirais.fn.butthp import butter_highpass
from moirais.fn._containers import SignalResult


def test_butthp_basic(signal_1khz):
    x, fs = signal_1khz
    result = butter_highpass(x, fs, cutoff=500)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_butthp_removes_low_freq(signal_1khz):
    x, fs = signal_1khz
    result = butter_highpass(x, fs, cutoff=500)
    fft_filt = np.abs(np.fft.rfft(result.filtered))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_1k = np.argmin(np.abs(freqs - 1000))
    assert fft_filt[idx_50] < fft_filt[idx_1k] * 0.1
