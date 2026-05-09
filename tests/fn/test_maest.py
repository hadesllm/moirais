"""Tests for maest.py - MA estimation."""
import numpy as np
from moirais.fn.maest import ma_estimate_fn, maest


def test_maest_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ma_estimate_fn(x, order=3)
    assert result.name == "ma_estimate"
    assert "ma_coeffs" in result.extra
    assert len(result.extra["ma_coeffs"]) == 3


def test_maest_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ma_estimate_fn(x)
    assert result.extra["sigma2"] > 0


def test_maest_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = maest(x, order=2)
    assert result.name == "ma_estimate"
