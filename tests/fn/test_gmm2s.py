"""Tests for moirais.fn.gmm2s -- Two-step GMM."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.gmm2s import gmm_twostep, gmm2s
from moirais.fn._containers import RegressionResult


class TestGMM2Step:
    def test_alias(self):
        assert gmm2s is gmm_twostep

    def test_just_identified(self):
        """With exactly as many instruments as regressors, GMM = IV."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.5 * z + 0.3 * u + rng.normal(0, 0.5, n)
        y = 1.0 + 2.0 * x + u
        df = pd.DataFrame({"outcome": y, "x": x, "z": z})
        result = gmm_twostep(df, x="x", z="z")
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["x"] - 2.0) < 1.0

    def test_overidentified(self):
        """Overidentified: 2 instruments for 1 endogenous."""
        rng = np.random.default_rng(42)
        n = 500
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.4 * z1 + 0.3 * z2 + 0.3 * u + rng.normal(0, 0.3, n)
        y = 1.0 + 2.0 * x + u
        df = pd.DataFrame({"outcome": y, "x": x, "z1": z1, "z2": z2})
        result = gmm_twostep(df, x="x", z=["z1", "z2"])
        assert abs(result.coefficients["x"] - 2.0) < 1.5
        assert "j_stat" in result.extra
        assert result.extra["j_df"] == 1

    def test_underidentified_raises(self):
        df = pd.DataFrame({"outcome": [1, 2], "x1": [1, 2], "x2": [3, 4], "z": [5, 6]})
        with pytest.raises(ValueError, match="instruments"):
            gmm_twostep(df, x=["x1", "x2"], z="z")
