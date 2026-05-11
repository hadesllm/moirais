"""Tests for morie.fn.gls -- Generalized least squares (Cochrane-Orcutt)."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.gls import generalized_ls, gls
from morie.fn._containers import RegressionResult


@pytest.fixture()
def ar1_data():
    """Data with AR(1) errors: y = 1 + 2*x + e, e_t = 0.7*e_{t-1} + u_t."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    # Generate AR(1) errors
    rho_true = 0.7
    u = rng.normal(0, 1, n)
    e = np.zeros(n)
    e[0] = u[0]
    for t in range(1, n):
        e[t] = rho_true * e[t - 1] + u[t]
    y = 1 + 2 * x + e
    return pd.DataFrame({"y": y, "x": x})


class TestGls:
    def test_alias(self):
        assert gls is generalized_ls

    def test_returns_regression_result(self, ar1_data):
        result = generalized_ls(ar1_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert "GLS" in result.method

    def test_rho_positive(self, ar1_data):
        """True rho=0.7, estimated should be positive."""
        result = generalized_ls(ar1_data, y="y", x="x")
        assert result.extra["rho"] > 0

    def test_rho_reasonable(self, ar1_data):
        """Estimated rho should be in the ballpark of 0.7."""
        result = generalized_ls(ar1_data, y="y", x="x")
        assert 0.3 < result.extra["rho"] < 0.95

    def test_slope_reasonable(self, ar1_data):
        """True slope is 2."""
        result = generalized_ls(ar1_data, y="y", x="x")
        assert abs(result.coefficients["x"] - 2.0) < 1.0

    def test_r_squared_positive(self, ar1_data):
        result = generalized_ls(ar1_data, y="y", x="x")
        assert result.r_squared > 0

    def test_n_correct(self, ar1_data):
        result = generalized_ls(ar1_data, y="y", x="x")
        assert result.n == 200
