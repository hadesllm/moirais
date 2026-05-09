"""Tests for manova_one."""
import numpy as np, pytest
from moirais.fn.manova import manova_one

class TestManova:
    def test_different(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (15, 2)), rng.normal(3, 1, (15, 2))])
        g = np.array([0]*15 + [1]*15)
        r = manova_one(X, g)
        assert r.statistic < 1.0

    def test_same(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 2))
        g = np.array([0]*15 + [1]*15)
        r = manova_one(X, g)
        assert r.test_name == "MANOVA"
