"""Tests for moirais.fn.baysl -- Bayesian LASSO."""

import numpy as np
import pytest
from moirais.fn.baysl import baysl


def _sim(n=60, p=10, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    beta_true = np.zeros(p)
    beta_true[:2] = [1.0, -0.5]
    y = Z @ beta_true + rng.standard_normal(n) * 0.5
    return y, Z


class TestBaysl:
    def test_returns_genomics_result(self):
        y, Z = _sim()
        res = baysl(y, Z, n_iter=200, burn_in=50)
        assert res.name == "BayesianLASSO"

    def test_effects_length(self):
        y, Z = _sim()
        res = baysl(y, Z, n_iter=200, burn_in=50)
        assert len(res.extra["effects"]) == Z.shape[1]

    def test_lambda_positive(self):
        y, Z = _sim()
        res = baysl(y, Z, n_iter=200, burn_in=50)
        assert res.extra["lambda_sq"] > 0

    def test_dimension_mismatch(self):
        y, Z = _sim()
        with pytest.raises(ValueError):
            baysl(y[:5], Z)
