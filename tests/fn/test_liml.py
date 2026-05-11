"""Tests for morie.fn.liml -- LIML estimator."""

import numpy as np
import pandas as pd
from morie.fn.liml import liml_estimator, liml
from morie.fn._containers import RegressionResult


class TestLIML:
    def test_alias(self):
        assert liml is liml_estimator

    def test_basic_iv(self):
        """LIML with strong instrument should recover coefficient."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.8 * z + 0.3 * u + rng.normal(0, 0.3, n)
        y = 1.0 + 2.5 * x + u
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = liml_estimator(df)
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["x_endog"] - 2.5) < 2.5
        assert "kappa" in result.extra

    def test_kappa_near_one_strong_iv(self):
        """With strong instruments, LIML kappa should be close to 1."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.normal(0, 1, n)
        x = 2.0 * z + rng.normal(0, 0.1, n)
        y = 3.0 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = liml_estimator(df)
        assert result.extra["kappa"] > 0.5
        assert abs(result.coefficients["x_endog"] - 3.0) < 1.0

    def test_se_positive(self):
        rng = np.random.default_rng(42)
        n = 200
        z = rng.normal(0, 1, n)
        x = z + rng.normal(0, 1, n)
        y = x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = liml_estimator(df)
        assert result.se["x_endog"] > 0
