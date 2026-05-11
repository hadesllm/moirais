"""Tests for burgp -- Burg AR spectral estimation."""
import numpy as np
from morie.fn.burgp import burg_psd
from morie.fn._containers import DescriptiveResult


def test_burg_basic(signal_1khz):
    x, fs = signal_1khz
    result = burg_psd(x, order=16, nfft=512, fs=fs)
    assert isinstance(result, DescriptiveResult)
    assert "frequencies" in result.extra
    assert "psd" in result.extra
    assert "ar_coeffs" in result.extra
    assert "noise_variance" in result.extra
    assert len(result.extra["psd"]) == 512


def test_burg_detects_sine():
    rng = np.random.default_rng(42)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t) + rng.standard_normal(len(t)) * 0.1
    result = burg_psd(x, order=32, nfft=1024, fs=fs)
    freqs = result.extra["frequencies"]
    psd = result.extra["psd"]
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 100) < 20


def test_burg_ar_coeffs_length():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(256)
    order = 8
    result = burg_psd(x, order=order, nfft=256, fs=1.0)
    assert len(result.extra["ar_coeffs"]) == order


def test_burg_noise_variance_positive():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(512)
    result = burg_psd(x, order=10, fs=1.0)
    assert result.extra["noise_variance"] > 0
