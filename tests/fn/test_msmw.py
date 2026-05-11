"""Tests for morie.fn.msmw — Marginal structural model weights."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.msmw import marginal_structural


class TestMarginalStructural:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-x))
        a = (rng.uniform(size=n) < ps).astype(int)
        df = pd.DataFrame({"treatment": a, "x": x})
        res = marginal_structural(df, treatment_col="treatment")
        assert res.extra["mean_weight"] > 0

    def test_ess_less_than_n(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "treatment": rng.choice([0, 1], n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })
        res = marginal_structural(df, treatment_col="treatment")
        assert res.extra["ess"] <= n + 1

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            marginal_structural(df, treatment_col="treatment")
