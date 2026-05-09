"""Tests for cooks_distance."""
import numpy as np, pytest
from moirais.fn.cooks import cooks_distance

class TestCooks:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 2))
        y = X[:, 0] + rng.normal(0, 0.1, 50)
        r = cooks_distance(X, y)
        assert r.name == "cooks_distance"
        assert len(r.extra["distances"]) == 50

    def test_outlier_flagged(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 1))
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        y[0] = 100
        r = cooks_distance(X, y)
        assert r.extra["n_influential"] >= 1
