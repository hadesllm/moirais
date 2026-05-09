"""Tests for moirais.fn.psbal -- Propensity score balancing weights."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.psbal import ps_balance, psbal


class TestPSBalance:
    def test_alias(self):
        assert psbal is ps_balance

    def test_weights_reduce_smd(self):
        """Weighting should reduce covariate imbalance."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        ps_true = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        t = rng.binomial(1, ps_true, n).astype(float)
        df = pd.DataFrame({"treatment": t, "x1": x1, "x2": x2})
        result = ps_balance(df, covariates=["x1", "x2"])
        assert "weights" in result
        assert "smd_before" in result
        assert "smd_after" in result
        for c in ["x1", "x2"]:
            assert abs(result["smd_after"][c]) <= abs(result["smd_before"][c]) + 0.1

    def test_with_precomputed_ps(self):
        rng = np.random.default_rng(42)
        n = 200
        ps = rng.uniform(0.1, 0.9, n)
        t = rng.binomial(1, ps, n).astype(float)
        df = pd.DataFrame({"treatment": t, "ps": ps})
        result = ps_balance(df, ps_col="ps")
        assert result["n"] == n
        assert len(result["weights"]) == n

    def test_requires_covariates_or_ps(self):
        df = pd.DataFrame({"treatment": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="covariates or ps_col"):
            ps_balance(df)
