"""Tests for morie.fn.arma — ARIMA model fitting."""
import numpy as np

from morie.fn.arma import arima_fit, arma


def test_ar1_coefficient_recovered():
    """AR(1) with phi=0.7 should recover coefficient roughly."""
    rng = np.random.default_rng(42)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.standard_normal()
    result = arima_fit(x, p=1, d=0, q=0)
    ar_coef = result.extra["ar"][0]
    assert abs(ar_coef - 0.7) < 0.2, f"AR coef={ar_coef}, expected ~0.7"
    assert result.extra["converged"]


def test_arima_has_aic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = arima_fit(x, p=1, d=0, q=0)
    assert "aic" in result.extra
    assert np.isfinite(result.extra["aic"])


def test_arma_alias():
    assert arma is arima_fit
