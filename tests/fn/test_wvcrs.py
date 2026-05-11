"""Tests for wvcrs.py - Cross-wavelet transform."""
import numpy as np
from morie.fn.wvcrs import wavelet_cross_spectrum, wvcrs


def test_cross_returns_result():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t) + rng.standard_normal(256) * 0.1
    y = np.sin(2 * np.pi * 10 * t + 0.5) + rng.standard_normal(256) * 0.1
    result = wavelet_cross_spectrum(x, y, scales=np.arange(1, 32))
    assert result.name == "wavelet_cross_spectrum"
    assert "cross_power" in result.extra


def test_cross_phase():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(128)
    y = rng.standard_normal(128)
    result = wavelet_cross_spectrum(x, y, scales=np.arange(1, 16))
    assert "cross_phase" in result.extra


def test_cross_alias():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    y = rng.standard_normal(64)
    result = wvcrs(x, y, scales=np.arange(1, 16))
    assert result.name == "wavelet_cross_spectrum"
