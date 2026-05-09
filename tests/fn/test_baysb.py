"""Tests for moirais.fn.baysb -- BayesB genomic prediction."""

import numpy as np
import pytest
from moirais.fn.baysb import baysb


def _sim(n=60, p=10, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    beta_true = np.zeros(p)
    beta_true[:2] = [1.5, -1.0]
    y = Z @ beta_true + rng.standard_normal(n) * 0.5
    return y, Z


class TestBaysb:
    def test_returns_genomics_result(self):
        y, Z = _sim()
        res = baysb(y, Z, n_iter=200, burn_in=50)
        assert res.name == "BayesB"

    def test_inclusion_prob_range(self):
        y, Z = _sim()
        res = baysb(y, Z, n_iter=200, burn_in=50)
        ip = np.array(res.extra["inclusion_prob"])
        assert np.all(ip >= 0) and np.all(ip <= 1)

    def test_sparsity(self):
        y, Z = _sim(p=20)
        res = baysb(y, Z, n_iter=300, burn_in=100, pi=0.9)
        ip = np.array(res.extra["inclusion_prob"])
        assert np.mean(ip) < 0.8

    def test_effects_length(self):
        y, Z = _sim()
        res = baysb(y, Z, n_iter=200, burn_in=50)
        assert len(res.extra["effects"]) == Z.shape[1]

    def test_dimension_mismatch(self):
        y, Z = _sim()
        with pytest.raises(ValueError):
            baysb(y[:5], Z)
