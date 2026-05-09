"""Tests for resac.py - Residual ACF."""
import numpy as np
from moirais.fn.resac import residual_acf_fn, resac


def test_resac_returns_result():
    residuals = np.random.default_rng(42).standard_normal(256)
    result = residual_acf_fn(residuals, nlags=10)
    assert result.name == "residual_acf"
    assert len(result.extra["acf"]) == 11


def test_resac_acf_at_lag0():
    residuals = np.random.default_rng(42).standard_normal(256)
    result = residual_acf_fn(residuals)
    assert abs(result.extra["acf"][0] - 1.0) < 1e-10


def test_resac_white_noise():
    residuals = np.random.default_rng(42).standard_normal(5000)
    result = residual_acf_fn(residuals, nlags=10)
    assert result.extra["white_noise"] is True


def test_resac_alias():
    residuals = np.random.default_rng(42).standard_normal(64)
    result = resac(residuals, nlags=5)
    assert result.name == "residual_acf"
