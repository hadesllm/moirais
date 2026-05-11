"""Tests for morie.fn.ebal -- Entropy balancing."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.ebal import entropy_balance, ebal


class TestEntropyBalance:
    def test_alias(self):
        assert ebal is entropy_balance

    def test_balances_covariates(self):
        """Entropy balancing should achieve good covariate balance."""
        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2)))
        t = rng.binomial(1, ps, n).astype(float)
        df = pd.DataFrame({"treatment": t, "x1": x1, "x2": x2})
        result = entropy_balance(df, covariates=["x1", "x2"])
        assert result["max_smd_after"] < 0.2
        assert result["n"] == n

    def test_treated_weights_are_one(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = (x > 0).astype(float)
        df = pd.DataFrame({"treatment": t, "x": x})
        result = entropy_balance(df, covariates=["x"])
        assert np.all(result["weights"][t == 1] == 1.0)

    def test_requires_covariates(self):
        df = pd.DataFrame({"treatment": [0, 1, 0, 1]})
        with pytest.raises(ValueError):
            entropy_balance(df)
