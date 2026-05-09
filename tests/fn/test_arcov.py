"""Tests for arcov.py - AR covariance estimation."""
import numpy as np
import pytest
from moirais.fn.arcov import ar_covariance_fn, arcov


def test_ar_covariance_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_covariance_fn(x)
    assert result.name == "ar_covariance"
    assert "coefficients" in result.extra
    assert "sigma2" in result.extra


def test_ar_covariance_coefficient_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_covariance_fn(x, order=8)
    assert len(result.extra["coefficients"]) == 8


def test_ar_covariance_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_covariance_fn(x)
    assert result.extra["sigma2"] > 0


def test_arcov_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = arcov(x, order=5)
    assert result.name == "ar_covariance"
