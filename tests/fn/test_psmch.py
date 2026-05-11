"""Tests for propensity_match."""
import numpy as np, pytest
from morie.fn.psmch import propensity_match

class TestPSMatch:
    def test_basic(self):
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        t = (X[:, 0] > 0).astype(float)
        r = propensity_match(t, X)
        assert r.extra["n_matched"] > 0

    def test_caliper(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (60, 1))
        t = (X[:, 0] > 0).astype(float)
        r = propensity_match(t, X, caliper=0.01)
        assert r.extra["n_matched"] <= int(t.sum())
