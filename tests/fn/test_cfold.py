"""Tests for morie.fn.cfold -- Cross-fitting for DML."""

import numpy as np
import pandas as pd
from morie.fn.cfold import cross_fit, cfold


class TestCrossFit:
    def test_alias(self):
        assert cfold is cross_fit

    def test_known_ate(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = cross_fit(df, covariates=["x"])
        assert abs(result["theta"] - 2.0) < 1.5
        assert result["se"] > 0

    def test_fold_estimates(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 1.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = cross_fit(df, covariates=["x"], n_folds=3)
        assert len(result["fold_estimates"]) == 3
        assert result["n_folds"] == 3

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        r1 = cross_fit(df, covariates=["x"], seed=99)
        r2 = cross_fit(df, covariates=["x"], seed=99)
        assert r1["theta"] == r2["theta"]
