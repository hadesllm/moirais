"""Tests for morie.fn.tsls -- Two-stage least squares."""

import numpy as np
import pandas as pd
from morie.fn.tsls import two_stage_ls, tsls
from morie.fn._containers import RegressionResult


class TestTSLS:
    def test_alias(self):
        assert tsls is two_stage_ls

    def test_recover_coefficient(self):
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.7 * z + 0.4 * u + rng.normal(0, 0.3, n)
        y = 1.0 + 2.0 * x + u
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = two_stage_ls(df)
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["x_endog"] - 2.0) < 1.5

    def test_first_stage_f(self):
        """First-stage F should be high with strong instrument."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = 2.0 * z + rng.normal(0, 0.5, n)
        y = 3.0 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = two_stage_ls(df)
        assert result.extra["first_stage_F"][0] > 10

    def test_hausman_test(self):
        """Hausman test should be in output."""
        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.5 * z + u
        y = 2.0 * x + u
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z})
        result = two_stage_ls(df)
        assert "hausman_stat" in result.extra
        assert "hausman_pval" in result.extra

    def test_with_exogenous_controls(self):
        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(0, 1, n)
        w = rng.normal(0, 1, n)
        x = 0.5 * z + rng.normal(0, 1, n)
        y = 2.0 * x + 1.0 * w + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "x_endog": x, "z": z, "w": w})
        result = two_stage_ls(df, x_exog=["w"])
        assert "w" in result.coefficients
