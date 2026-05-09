"""Tests for moirais.fn.qreg -- Quantile regression."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.qreg import quantile_regression, qreg
from moirais.fn._containers import RegressionResult


@pytest.fixture()
def qreg_data():
    """Simple linear data: y = 1 + 2*x + noise."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x})


class TestQreg:
    def test_alias(self):
        assert qreg is quantile_regression

    def test_returns_regression_result(self, qreg_data):
        result = quantile_regression(qreg_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert "Quantile" in result.method

    def test_median_similar_to_ols(self, qreg_data):
        """Median regression (tau=0.5) should give similar results to OLS for symmetric errors."""
        from moirais.fn.rey import linear_regression
        ols = linear_regression(qreg_data, y="y", x="x")
        med = quantile_regression(qreg_data, y="y", x="x", tau=0.5)
        # Slopes should be within 0.5 of each other
        assert abs(med.coefficients["x"] - ols.coefficients["x"]) < 0.5

    def test_tau_in_extra(self, qreg_data):
        result = quantile_regression(qreg_data, y="y", x="x", tau=0.75)
        assert result.extra["tau"] == 0.75

    def test_slope_positive(self, qreg_data):
        result = quantile_regression(qreg_data, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_n_correct(self, qreg_data):
        result = quantile_regression(qreg_data, y="y", x="x")
        assert result.n == 200
