"""Tests for dwtfn.py - Discrete Wavelet Transform."""
import numpy as np
from moirais.fn.dwtfn import dwt_decompose, dwtfn


def test_dwt_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = dwt_decompose(x)
    assert result.name == "dwt_decompose"
    assert "coeffs" in result.extra


def test_dwt_coeffs_structure():
    x = np.random.default_rng(42).standard_normal(256)
    result = dwt_decompose(x, wavelet="db4", level=3)
    coeffs = result.extra["coeffs"]
    assert len(coeffs) == 4


def test_dwt_haar():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = dwt_decompose(x, wavelet="haar", level=1)
    assert result.extra["level"] == 1
    assert len(result.extra["coeffs"]) == 2


def test_dwt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = dwtfn(x)
    assert result.name == "dwt_decompose"
