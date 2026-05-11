"""Tests for morie.fn.rey_ps -- Poisson regression."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.rey_ps import poisson_regression, rey_ps
from morie.fn._containers import RegressionResult


@pytest.fixture()
def poisson_data():
    """Count data: log(mu) = 0.5 + 0.3*x."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x": x})


class TestReyPs:
    def test_alias(self):
        assert rey_ps is poisson_regression

    def test_returns_regression_result(self, poisson_data):
        result = poisson_regression(poisson_data, y="y", x="x")
        assert isinstance(result, RegressionResult)
        assert result.method == "Poisson"

    def test_slope_positive(self, poisson_data):
        """True coefficient is +0.3, should be positive."""
        result = poisson_regression(poisson_data, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_slope_reasonable(self, poisson_data):
        """Slope should be in the neighborhood of 0.3."""
        result = poisson_regression(poisson_data, y="y", x="x")
        assert abs(result.coefficients["x"] - 0.3) < 0.2

    def test_intercept_reasonable(self, poisson_data):
        """Intercept should be near 0.5."""
        result = poisson_regression(poisson_data, y="y", x="x")
        assert abs(result.coefficients["(Intercept)"] - 0.5) < 0.3

    def test_fitted_positive(self, poisson_data):
        result = poisson_regression(poisson_data, y="y", x="x")
        assert np.all(result.fitted > 0)

    def test_n_correct(self, poisson_data):
        result = poisson_regression(poisson_data, y="y", x="x")
        assert result.n == 200
