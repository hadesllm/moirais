"""Tests for wvscl.py - Wavelet scalogram."""
import numpy as np
from morie.fn.wvscl import wavelet_scalogram, wvscl


def test_scalogram_returns_result():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t)
    result = wavelet_scalogram(x, fs=256.0)
    assert result.name == "wavelet_scalogram"
    assert "scalogram" in result.extra


def test_scalogram_shape():
    x = np.random.default_rng(42).standard_normal(128)
    scales = np.arange(1, 32)
    result = wavelet_scalogram(x, scales=scales)
    assert result.extra["scalogram"].shape == (31, 128)


def test_scalogram_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvscl(x, scales=np.arange(1, 16))
    assert result.name == "wavelet_scalogram"
