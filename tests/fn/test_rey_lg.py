"""Tests for moirais.fn.rey_lg -- Logistic regression via IRLS."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.rey_lg import logistic_regression, rey_lg
from moirais.fn._containers import RegressionResult


@pytest.fixture()
def logistic_data():
    """Binary outcome from logistic model: logit(p) = -1 + 2*x."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    from scipy.special import expit
    p = expit(-1 + 2 * x)
    y = rng.binomial(1, p, n).astype(float)
    return pd.DataFrame({"y": y, "x": x})


class TestReyLg:
    def test_alias(self):
        assert rey_lg is logistic_regression

    def test_returns_regression_result(self, logistic_data):
        result = logistic_regression(logistic_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert "Logistic" in result.method

    def test_slope_positive(self, logistic_data):
        """True slope is +2, estimate should be positive."""
        result = logistic_regression(logistic_data, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_slope_reasonable(self, logistic_data):
        """Slope should be in ballpark of 2."""
        result = logistic_regression(logistic_data, y="y", x="x")
        assert 1.0 < result.coefficients["x"] < 4.0

    def test_intercept_negative(self, logistic_data):
        """True intercept is -1, estimate should be negative."""
        result = logistic_regression(logistic_data, y="y", x="x")
        assert result.coefficients["(Intercept)"] < 0

    def test_pseudo_r2_positive(self, logistic_data):
        result = logistic_regression(logistic_data, y="y", x="x")
        assert result.r_squared > 0

    def test_log_likelihood_in_extra(self, logistic_data):
        result = logistic_regression(logistic_data, y="y", x="x")
        assert "log_likelihood" in result.extra
        assert result.extra["log_likelihood"] < 0

    def test_p_values_significant(self, logistic_data):
        result = logistic_regression(logistic_data, y="y", x="x")
        assert result.p_values["x"] < 0.05
