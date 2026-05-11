"""Tests for hurdle_model."""
import numpy as np, pytest
from morie.fn.hurld import hurdle_model

class TestHurdle:
    def test_basic(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(20), rng.poisson(3, 80)])
        X = rng.normal(0, 1, (100, 1))
        r = hurdle_model(y, X)
        assert r.name == "hurdle"
        assert r.extra["n_zeros"] >= 20

    def test_ll_finite(self):
        rng = np.random.default_rng(1)
        y = rng.poisson(2, 60).astype(float)
        X = rng.normal(0, 1, (60, 1))
        r = hurdle_model(y, X)
        assert np.isfinite(r.extra["log_likelihood"])
