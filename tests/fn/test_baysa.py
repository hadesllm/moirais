"""Tests for morie.fn.baysa -- BayesA genomic prediction."""

import numpy as np
import pytest
from morie.fn.baysa import baysa


def _sim(n=60, p=10, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    beta_true = np.zeros(p)
    beta_true[:3] = [1.0, -0.5, 0.8]
    y = Z @ beta_true + rng.standard_normal(n) * 0.5
    return y, Z


class TestBaysa:
    def test_returns_genomics_result(self):
        y, Z = _sim()
        res = baysa(y, Z, n_iter=200, burn_in=50)
        assert res.name == "BayesA"

    def test_effects_length(self):
        y, Z = _sim()
        res = baysa(y, Z, n_iter=200, burn_in=50)
        assert len(res.extra["effects"]) == Z.shape[1]

    def test_positive_correlation(self):
        y, Z = _sim()
        res = baysa(y, Z, n_iter=500, burn_in=100)
        assert res.statistic > 0.0

    def test_var_e_positive(self):
        y, Z = _sim()
        res = baysa(y, Z, n_iter=200, burn_in=50)
        assert res.extra["var_e"] > 0

    def test_dimension_mismatch(self):
        y, Z = _sim()
        with pytest.raises(ValueError):
            baysa(y[:5], Z)
