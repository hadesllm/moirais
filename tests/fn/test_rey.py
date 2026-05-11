"""Tests for morie.fn.rey -- OLS linear regression."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.rey import linear_regression, rey
from morie.fn._containers import RegressionResult


@pytest.fixture()
def ols_data():
    """y = 0.5 + 2*x + noise."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x})


class TestRey:
    def test_alias(self):
        assert rey is linear_regression

    def test_returns_regression_result(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert result.method == "OLS"

    def test_intercept_close_to_half(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert abs(result.coefficients["(Intercept)"] - 0.5) < 0.2

    def test_slope_close_to_two(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert abs(result.coefficients["x"] - 2.0) < 0.2

    def test_r_squared_high(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert result.r_squared > 0.8

    def test_p_values_significant(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert result.p_values["x"] < 0.001

    def test_n_and_k(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        assert result.n == 200
        assert result.k == 1

    def test_multiple_predictors(self):
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        result = linear_regression(df, y="y", x=["x1", "x2"])
        assert result.k == 2
        assert "(Intercept)" in result.coefficients

    def test_summary_string(self, ols_data):
        result = linear_regression(ols_data, y="y", x="x")
        s = result.summary()
        assert "OLS" in s
        assert "R" in s

    def test_missing_column_raises(self, ols_data):
        with pytest.raises(ValueError, match="Missing columns"):
            linear_regression(ols_data, y="y", x="nonexistent")
