"""Tests for armae.py - ARMA estimation."""
import numpy as np
from moirais.fn.armae import arma_estimate_fn, armae


def test_armae_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = arma_estimate_fn(x, p=3, q=1)
    assert result.name == "arma_estimate"
    assert "ar_coeffs" in result.extra
    assert len(result.extra["ar_coeffs"]) == 3


def test_armae_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = arma_estimate_fn(x)
    assert result.extra["sigma2"] > 0


def test_armae_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = armae(x, p=2, q=1)
    assert result.name == "arma_estimate"
