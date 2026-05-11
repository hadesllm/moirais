"""Tests for stmcb.py - Steiglitz-McBride ARMA estimation."""
import numpy as np
from morie.fn.stmcb import steiglitz_mcbride_fn, stmcb


def test_stmcb_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = steiglitz_mcbride_fn(x)
    assert result.name == "steiglitz_mcbride"
    assert "ar_coeffs" in result.extra
    assert "ma_coeffs" in result.extra


def test_stmcb_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = steiglitz_mcbride_fn(x, p=5, q=2)
    assert len(result.extra["ar_coeffs"]) == 5


def test_stmcb_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = steiglitz_mcbride_fn(x)
    assert result.extra["sigma2"] > 0


def test_stmcb_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = stmcb(x, p=2, q=1)
    assert result.name == "steiglitz_mcbride"
