"""Tests for morie.fn.iptw -- Stabilized IPTW."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.iptw import stabilized_iptw, iptw
from morie.fn._containers import ESRes


class TestStabilizedIPTW:
    def test_alias(self):
        assert iptw is stabilized_iptw

    def test_known_ate(self):
        """Recover known ATE of 2.0."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-0.5 * x))
        t = rng.binomial(1, ps, n).astype(float)
        y = 1.0 + 2.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = stabilized_iptw(df, covariates=["x"])
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 2.0) < 1.0
        assert result.se > 0

    def test_with_precomputed_ps(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-x))
        t = rng.binomial(1, ps, n).astype(float)
        y = 3.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "ps": ps})
        result = stabilized_iptw(df, ps_col="ps")
        assert result.extra["ess_treated"] > 0
        assert result.extra["ess_control"] > 0

    def test_ess_smaller_than_n(self):
        """Effective sample size should be <= actual n per group."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        ps = np.clip(1 / (1 + np.exp(-2 * x)), 0.05, 0.95)
        t = rng.binomial(1, ps, n).astype(float)
        y = t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "ps": ps})
        result = stabilized_iptw(df, ps_col="ps")
        assert result.extra["ess_treated"] <= (t == 1).sum()
        assert result.extra["ess_control"] <= (t == 0).sum()
