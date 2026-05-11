"""Tests for wvvar.py - Wavelet variance."""
import numpy as np
from morie.fn.wvvar import wavelet_variance, wvvar


def test_wvvar_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_variance(x, wavelet="haar", level=2)
    assert result.name == "wavelet_variance"
    assert "variances" in result.extra


def test_wvvar_positive_variances():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_variance(x, wavelet="haar")
    assert all(v >= 0 for v in result.extra["variances"])


def test_wvvar_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvvar(x, wavelet="haar")
    assert result.name == "wavelet_variance"
