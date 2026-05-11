"""Tests for lpcco.py - LPC coefficients."""
import numpy as np
from morie.fn.lpcco import lpc_coefficients_fn, lpcco


def test_lpcco_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = lpc_coefficients_fn(x, order=10)
    assert result.name == "lpc_coefficients"
    assert len(result.extra["coefficients"]) == 10


def test_lpcco_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = lpc_coefficients_fn(x)
    assert result.extra["sigma2"] > 0


def test_lpcco_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = lpcco(x, order=5)
    assert result.name == "lpc_coefficients"
