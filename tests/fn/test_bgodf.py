"""Tests for moirais.fn.bgodf — Breusch-Godfrey serial correlation LM test."""

import numpy as np
import pytest

from moirais.fn.bgodf import bg_test, bgodf


def _ar1_resid(phi: float, n: int, seed: int = 0) -> tuple:
    """Return (y, x) where y = phi*y[t-1] + N(0,1) errors on x=const."""
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    eps[0] = rng.standard_normal()
    for i in range(1, n):
        eps[i] = phi * eps[i - 1] + rng.standard_normal()
    x = np.ones(n)
    return eps, x


def test_returns_test_result():
    """Return type has the standard TestResult interface."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    r = bg_test(y, lags=2)
    assert hasattr(r, "statistic")
    assert hasattr(r, "p_value")
    assert 0.0 <= r.p_value <= 1.0
    assert r.df == 2.0


def test_white_noise_not_rejected():
    """i.i.d. residuals — BG test should NOT reject (p > 0.05)."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)
    r = bg_test(y, lags=4)
    assert r.p_value > 0.05, f"Expected p > 0.05 for white noise, got {r.p_value}"


def test_serial_correlation_detected():
    """Strong AR(1) residuals — BG test should reject (p < 0.05)."""
    y, _ = _ar1_resid(0.8, 200, seed=1)
    r = bg_test(y, lags=4)
    assert r.p_value < 0.05, f"Expected p < 0.05 for AR(1) resid, got {r.p_value}"


def test_multiple_lags():
    """Multiple lags accepted."""
    rng = np.random.default_rng(5)
    y = rng.standard_normal(100)
    for lags in (1, 2, 4):
        r = bg_test(y, lags=lags)
        assert r.df == float(lags)


def test_with_exogenous():
    """With exogenous regressors."""
    rng = np.random.default_rng(7)
    y = rng.standard_normal(80)
    x = rng.standard_normal((80, 2))
    r = bg_test(y, lags=2, x=x)
    assert 0.0 <= r.p_value <= 1.0


def test_lm_statistic_nonneg():
    """LM statistic n*R^2 is always >= 0."""
    rng = np.random.default_rng(9)
    y = rng.standard_normal(60)
    r = bg_test(y, lags=3)
    assert r.statistic >= 0.0


def test_invalid_lags_raises():
    with pytest.raises(ValueError):
        bg_test([1.0] * 20, lags=0)


def test_too_short_raises():
    with pytest.raises(ValueError):
        bg_test([1.0, 2.0, 3.0, 4.0], lags=4)


def test_alias():
    assert bgodf is bg_test
