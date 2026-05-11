"""Tests for morie.fn.dbias -- Double debiasing."""

import numpy as np
import pandas as pd
from morie.fn.dbias import double_debias, dbias
from morie.fn._containers import ESRes


class TestDoubleDebias:
    def test_alias(self):
        assert dbias is double_debias

    def test_known_ate(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + 0.8 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = double_debias(df, covariates=["x"])
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 2.0) < 1.5

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 1.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = double_debias(df, covariates=["x"])
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_cross_fitting(self):
        rng = np.random.default_rng(42)
        n = 400
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 3.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = double_debias(df, covariates=["x"], n_folds=5)
        assert result.extra["n_folds"] == 5
