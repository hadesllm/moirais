"""Tests for armnr.py - ARMA Newton-Raphson estimation."""
import numpy as np
from moirais.fn.armnr import arma_newton_raphson_fn, armnr


def test_armnr_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = arma_newton_raphson_fn(x)
    assert result.name == "arma_newton_raphson"
    assert "ar_coeffs" in result.extra
    assert "ma_coeffs" in result.extra


def test_armnr_coefficient_counts():
    x = np.random.default_rng(42).standard_normal(256)
    result = arma_newton_raphson_fn(x, p=3, q=1)
    assert len(result.extra["ar_coeffs"]) == 3
    assert len(result.extra["ma_coeffs"]) == 2


def test_armnr_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = arma_newton_raphson_fn(x)
    assert result.extra["sigma2"] >= 0


def test_armnr_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = armnr(x, p=2, q=1)
    assert result.name == "arma_newton_raphson"
