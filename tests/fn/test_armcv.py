"""Tests for armcv.py - AR modified covariance method."""
import numpy as np
from morie.fn.armcv import ar_modified_cov_fn, armcv


def test_armcv_returns_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_modified_cov_fn(x, order=4)
    assert result.name == "ar_modified_covariance"
    assert "coefficients" in result.extra
    assert len(result.extra["coefficients"]) == 4


def test_armcv_sigma2_positive():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_modified_cov_fn(x)
    assert result.extra["sigma2"] > 0


def test_armcv_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = armcv(x, order=2)
    assert result.name == "ar_modified_covariance"
