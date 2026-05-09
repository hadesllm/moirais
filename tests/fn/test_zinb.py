"""Tests for zero_inflated_negbin."""
import numpy as np, pytest
from moirais.fn.zinb import zero_inflated_negbin

class TestZINB:
    def test_basic(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(30), rng.negative_binomial(3, 0.3, 70)])
        X = rng.normal(0, 1, (100, 1))
        r = zero_inflated_negbin(y, X)
        assert r.name == "zinb"
        assert r.extra["zero_prob"] > 0

    def test_ll_finite(self):
        rng = np.random.default_rng(1)
        y = rng.poisson(2, 80).astype(float)
        y[:10] = 0
        X = rng.normal(0, 1, (80, 1))
        r = zero_inflated_negbin(y, X)
        assert np.isfinite(r.extra["log_likelihood"])
