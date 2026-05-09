"""Tests for wvdst -- Wigner-Ville distribution."""
import numpy as np
from moirais.fn.wvdst import wigner_ville
from moirais.fn._containers import DescriptiveResult


def test_wvd_basic():
    fs = 256
    t = np.arange(0, 0.5, 1 / fs)
    x = np.sin(2 * np.pi * 30 * t)
    result = wigner_ville(x, fs=fs)
    assert isinstance(result, DescriptiveResult)
    assert "wvd" in result.extra
    assert "times" in result.extra
    assert "frequencies" in result.extra
    wvd = result.extra["wvd"]
    assert wvd.shape[1] == len(x)


def test_wvd_energy_at_signal_freq():
    fs = 256
    t = np.arange(0, 1.0, 1 / fs)
    f0 = 40
    x = np.sin(2 * np.pi * f0 * t)
    result = wigner_ville(x, fs=fs)
    freqs = result.extra["frequencies"]
    wvd = result.extra["wvd"]
    marginal = np.sum(np.abs(wvd), axis=1)
    peak_freq = freqs[np.argmax(marginal)]
    assert abs(peak_freq - f0) < 10


def test_wvd_custom_nfft():
    x = np.sin(2 * np.pi * 10 * np.arange(64) / 64.0)
    result = wigner_ville(x, fs=64, nfft=128)
    assert result.extra["wvd"].shape[0] == 64
