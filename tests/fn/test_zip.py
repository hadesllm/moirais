"""Tests for zero_inflated_poisson."""
import numpy as np, pytest
from moirais.fn.zip import zero_inflated_poisson

class TestZIP:
    def test_basic(self):
        rng = np.random.default_rng(0)
        n = 100
        y = rng.poisson(2, n)
        y[:20] = 0
        X = rng.normal(0, 1, (n, 1))
        r = zero_inflated_poisson(y, X)
        assert r.name == "zip"
        assert r.extra["zero_prob"] > 0

    def test_n_zeros(self):
        rng = np.random.default_rng(1)
        y = np.concatenate([np.zeros(30), rng.poisson(3, 70)])
        X = rng.normal(0, 1, (100, 1))
        r = zero_inflated_poisson(y, X)
        assert r.extra["n_zeros"] >= 25
