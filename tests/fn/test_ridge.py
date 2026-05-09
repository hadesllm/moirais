"""Tests for moirais.fn.ridge -- Ridge regression (L2)."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.ridge import ridge_regression, ridge
from moirais.fn._containers import RegressionResult


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(0, 1, n)
    y = 1 + 3 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x})


class TestRidge:
    def test_alias(self):
        assert ridge is ridge_regression

    def test_returns_regression_result(self, reg_data):
        result = ridge_regression(reg_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert "Ridge" in result.method

    def test_r_squared_positive(self, reg_data):
        result = ridge_regression(reg_data, y="y", x="x")
        assert result.r_squared > 0

    def test_coefficients_finite(self, reg_data):
        result = ridge_regression(reg_data, y="y", x="x")
        assert np.isfinite(result.coefficients["x"])
        assert np.isfinite(result.coefficients["(Intercept)"])

    def test_slope_shrunk_toward_zero(self, reg_data):
        """Ridge should shrink slope vs OLS (with large lambda)."""
        from moirais.fn.rey import linear_regression
        ols = linear_regression(reg_data, y="y", x="x")
        ridge_res = ridge_regression(reg_data, y="y", x="x", lam=10.0)
        assert abs(ridge_res.coefficients["x"]) < abs(ols.coefficients["x"])

    def test_lambda_in_extra(self, reg_data):
        result = ridge_regression(reg_data, y="y", x="x", lam=2.0)
        assert result.extra["lambda"] == 2.0
        assert "df_effective" in result.extra

    def test_multiple_predictors(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "y": rng.normal(0, 1, 100),
            "x1": rng.normal(0, 1, 100),
            "x2": rng.normal(0, 1, 100),
        })
        result = ridge_regression(df, y="y", x=["x1", "x2"])
        assert result.k == 2
