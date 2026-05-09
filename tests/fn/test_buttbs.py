"""Tests for buttbs — Butterworth bandstop filter."""
import numpy as np
from moirais.fn.buttbs import butter_bandstop
from moirais.fn._containers import SignalResult


def test_buttbs_basic(signal_1khz):
    x, fs = signal_1khz
    result = butter_bandstop(x, fs, low=40, high=60)
    assert isinstance(result, SignalResult)
    assert result.n_samples == len(x)


def test_buttbs_removes_50hz(signal_1khz):
    x, fs = signal_1khz
    result = butter_bandstop(x, fs, low=40, high=60)
    fft_orig = np.abs(np.fft.rfft(x))
    fft_filt = np.abs(np.fft.rfft(result.filtered))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    idx_50 = np.argmin(np.abs(freqs - 50))
    assert fft_filt[idx_50] < fft_orig[idx_50] * 0.1
