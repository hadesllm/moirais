"""Tests for wvcor.py - Wavelet correlation."""
import numpy as np
from moirais.fn.wvcor import wavelet_correlation, wvcor


def test_wvcor_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(128)
    y = x + 0.1 * rng.standard_normal(128)
    result = wavelet_correlation(x, y, wavelet="haar", level=2)
    assert result.name == "wavelet_correlation"
    assert "correlations" in result.extra


def test_wvcor_identical_signals():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_correlation(x, x, wavelet="haar", level=2)
    for r in result.extra["correlations"]:
        np.testing.assert_allclose(r, 1.0, atol=1e-10)


def test_wvcor_alias():
    x = np.random.default_rng(42).standard_normal(64)
    y = np.random.default_rng(43).standard_normal(64)
    result = wvcor(x, y, wavelet="haar", level=1)
    assert result.name == "wavelet_correlation"
