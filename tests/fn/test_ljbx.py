"""Tests for ljbx.py - Ljung-Box test."""
import numpy as np
from moirais.fn.ljbx import ljung_box_test_fn, ljbx


def test_ljbx_returns_result():
    residuals = np.random.default_rng(42).standard_normal(256)
    result = ljung_box_test_fn(residuals, nlags=10)
    assert result.name == "ljung_box"
    assert "Q" in result.extra
    assert "p_value" in result.extra


def test_ljbx_white_noise_high_pvalue():
    residuals = np.random.default_rng(42).standard_normal(1000)
    result = ljung_box_test_fn(residuals, nlags=10)
    assert result.extra["p_value"] > 0.01


def test_ljbx_alias():
    residuals = np.random.default_rng(42).standard_normal(64)
    result = ljbx(residuals, nlags=5)
    assert result.name == "ljung_box"
