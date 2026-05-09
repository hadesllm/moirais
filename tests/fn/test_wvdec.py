"""Tests for wvdec.py - Wavelet decompose."""
import numpy as np
from moirais.fn.wvdec import wavelet_decompose, wvdec


def test_decompose_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_decompose(x, level=3)
    assert result.name == "wavelet_decompose"
    assert "coeffs" in result.extra


def test_decompose_coeffs_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_decompose(x, level=3)
    assert len(result.extra["coeffs"]) == 4


def test_decompose_haar():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    result = wavelet_decompose(x, wavelet="haar", level=2)
    assert result.extra["level"] == 2


def test_decompose_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvdec(x, level=2)
    assert result.name == "wavelet_decompose"
