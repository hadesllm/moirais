"""Tests for morie.fn.nuis -- Nuisance parameter estimation."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.nuis import nuisance_estimate, nuis


class TestNuisance:
    def test_alias(self):
        assert nuis is nuisance_estimate

    def test_m_hat_shape(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = nuisance_estimate(df, covariates=["x"])
        assert len(result["m_hat"]) == n
        assert len(result["e_hat"]) == n

    def test_e_hat_bounded(self):
        """Propensity scores should be clipped to [0.01, 0.99]."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = (x > 0).astype(float)
        y = t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = nuisance_estimate(df, covariates=["x"])
        assert result["e_hat"].min() >= 0.01
        assert result["e_hat"].max() <= 0.99

    def test_requires_covariates(self):
        df = pd.DataFrame({"outcome": [1], "treatment": [0]})
        with pytest.raises(ValueError):
            nuisance_estimate(df)

    def test_ols_method(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = nuisance_estimate(df, covariates=["x"], method="ols")
        assert result["n"] == n
