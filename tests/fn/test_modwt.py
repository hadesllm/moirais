"""Tests for modwt.py - MODWT."""
import numpy as np
from morie.fn.modwt import modwt_decompose, modwt


def test_modwt_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = modwt_decompose(x, wavelet="haar", level=3)
    assert result.name == "modwt_decompose"
    assert "coeffs" in result.extra


def test_modwt_same_length():
    x = np.random.default_rng(42).standard_normal(128)
    result = modwt_decompose(x, wavelet="haar", level=2)
    for ca, cd in result.extra["coeffs"]:
        assert len(ca) == len(x)
        assert len(cd) == len(x)


def test_modwt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = modwt(x, wavelet="haar", level=1)
    assert result.name == "modwt_decompose"
