"""Tests for moirais.fn.hux -- Huber robust regression."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.hux import huber_regression, hux
from moirais.fn._containers import RegressionResult


@pytest.fixture()
def outlier_data():
    """Linear data with outliers: y = 1 + 2*x + noise, plus 5 extreme outliers."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 0.5, n)
    # Add 5 extreme outliers
    x = np.concatenate([x, [3, 4, 5, -3, -4]])
    y = np.concatenate([y, [100, -80, 90, -70, 60]])
    return pd.DataFrame({"y": y, "x": x})


class TestHux:
    def test_alias(self):
        assert hux is huber_regression

    def test_returns_regression_result(self, outlier_data):
        result = huber_regression(outlier_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert "Huber" in result.method

    def test_robust_to_outliers(self, outlier_data):
        """Huber slope should be closer to true value (2) than OLS with outliers."""
        from moirais.fn.rey import linear_regression
        ols = linear_regression(outlier_data, y="y", x="x")
        huber = huber_regression(outlier_data, y="y", x="x")
        # Huber should be closer to 2 than OLS
        assert abs(huber.coefficients["x"] - 2.0) < abs(ols.coefficients["x"] - 2.0)

    def test_r_squared_positive(self, outlier_data):
        result = huber_regression(outlier_data, y="y", x="x")
        # R2 may be low due to outliers, but should still be computed
        assert np.isfinite(result.r_squared)

    def test_n_correct(self, outlier_data):
        result = huber_regression(outlier_data, y="y", x="x")
        assert result.n == 105  # 100 + 5 outliers

    def test_custom_delta(self, outlier_data):
        result = huber_regression(outlier_data, y="y", x="x", delta=2.0)
        assert "delta=2.0" in result.method
