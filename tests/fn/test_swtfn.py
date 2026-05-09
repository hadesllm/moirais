"""Tests for swtfn.py - Stationary Wavelet Transform."""
import numpy as np
from moirais.fn.swtfn import swt_decompose, swtfn


def test_swt_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = swt_decompose(x, level=2)
    assert result.name == "swt_decompose"
    assert "coeffs" in result.extra


def test_swt_same_length():
    x = np.random.default_rng(42).standard_normal(128)
    result = swt_decompose(x, wavelet="haar", level=3)
    for ca, cd in result.extra["coeffs"]:
        assert len(ca) == len(x)
        assert len(cd) == len(x)


def test_swt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = swtfn(x, level=1)
    assert result.name == "swt_decompose"
