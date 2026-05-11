"""Tests for xwvlt.py - Cross-wavelet spectrum."""
import numpy as np
from morie.fn.xwvlt import cross_wavelet, xwvlt


def test_xwvlt_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    y = np.cos(np.linspace(0, 4 * np.pi, 128))
    result = cross_wavelet(x, y, fs=128.0)
    assert result.name == "cross_wavelet"
    assert "cross_spectrum" in result.extra
    assert "power" in result.extra
    assert "phase" in result.extra


def test_xwvlt_power_positive():
    x = np.sin(np.linspace(0, 4 * np.pi, 64))
    y = np.cos(np.linspace(0, 4 * np.pi, 64))
    result = cross_wavelet(x, y)
    assert np.all(result.extra["power"] >= 0)


def test_xwvlt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    y = np.random.default_rng(43).standard_normal(64)
    result = xwvlt(x, y)
    assert result.name == "cross_wavelet"
