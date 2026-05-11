"""Tests for eegar -- EEG autoregressive modeling."""
import numpy as np
from morie.fn.eegar import eegar
from morie.fn._containers import DescriptiveResult


def test_eegar_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = eegar(x, fs=256.0, order=8)
    assert isinstance(result, DescriptiveResult)
    assert "psd" in result.extra
    assert "ar_coeffs" in result.extra


def test_eegar_ar_length():
    x = np.random.default_rng(7).standard_normal(256)
    result = eegar(x, order=10)
    assert len(result.extra["ar_coeffs"]) == 10


def test_eegar_detects_tone():
    fs = 256
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t)
    result = eegar(x, fs=fs, order=16, nfft=512)
    freqs = result.extra["frequencies"]
    psd = result.extra["psd"]
    peak = freqs[np.argmax(psd)]
    assert abs(peak - 10) < 5
