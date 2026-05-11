"""Tests for mywkr.py - Modified Yule-Walker ARMA estimation."""
import numpy as np
from morie.fn.mywkr import modified_yule_walker_arma_fn, mywkr


def test_mywkr_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = modified_yule_walker_arma_fn(x)
    assert result.name == "modified_yw_arma"
    assert "ar_coeffs" in result.extra


def test_mywkr_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = modified_yule_walker_arma_fn(x, p=6, q=2)
    assert len(result.extra["ar_coeffs"]) == 6


def test_mywkr_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = modified_yule_walker_arma_fn(x)
    assert result.extra["sigma2"] > 0


def test_mywkr_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = mywkr(x, p=3, q=1)
    assert result.name == "modified_yw_arma"
