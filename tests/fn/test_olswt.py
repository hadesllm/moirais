"""Tests for morie.fn.olswt -- OLS with IPW weights."""

import numpy as np
import pandas as pd
from morie.fn.olswt import ols_weighted, olswt
from morie.fn._containers import RegressionResult


class TestOLSWeighted:
    def test_alias(self):
        assert olswt is ols_weighted

    def test_known_ate(self):
        rng = np.random.default_rng(42)
        n = 400
        x = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-x))
        t = rng.binomial(1, ps, n).astype(float)
        y = 1.0 + 3.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
        df = pd.DataFrame({"outcome": y, "treatment": t, "weight": w, "x": x})
        result = ols_weighted(df)
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["treatment"] - 3.0) < 1.5
        assert result.se["treatment"] > 0

    def test_with_covariates(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "weight": np.ones(n), "x": x})
        result = ols_weighted(df, x=["x"])
        assert "x" in result.coefficients
        assert abs(result.coefficients["x"] - 1.0) < 1.0

    def test_r_squared_between_0_and_1(self):
        rng = np.random.default_rng(42)
        n = 100
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 5.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "weight": np.ones(n)})
        result = ols_weighted(df)
        assert 0 <= result.r_squared <= 1
