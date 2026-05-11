"""Tests for psdwl -- Welch PSD."""
import numpy as np
from morie.fn.psdwl import psdwl
from morie.fn._containers import DescriptiveResult


def test_psdwl_basic(signal_1khz):
    x, fs = signal_1khz
    result = psdwl(x, fs)
    assert isinstance(result, DescriptiveResult)
    assert "psd" in result.extra
    assert "frequencies" in result.extra


def test_psdwl_peak_at_signal():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t) + rng.standard_normal(len(t)) * 0.1
    result = psdwl(x, fs, nperseg=256)
    freqs = result.extra["frequencies"]
    psd = result.extra["psd"]
    peak = freqs[np.argmax(psd)]
    assert abs(peak - 100) < 20


def test_psdwl_total_power_positive():
    x = np.random.default_rng(7).standard_normal(512)
    result = psdwl(x, 1.0)
    assert result.extra["total_power"] > 0
