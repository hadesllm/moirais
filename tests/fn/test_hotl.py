"""Tests for hotelling_t2."""
import numpy as np, pytest
from morie.fn.hotl import hotelling_t2

class TestHotelling:
    def test_different(self):
        rng = np.random.default_rng(0)
        X1 = rng.normal(0, 1, (20, 2))
        X2 = rng.normal(3, 1, (20, 2))
        r = hotelling_t2(X1, X2)
        assert r.p_value < 0.05

    def test_same(self):
        rng = np.random.default_rng(1)
        X1 = rng.normal(0, 1, (20, 2))
        X2 = rng.normal(0, 1, (20, 2))
        r = hotelling_t2(X1, X2)
        assert r.test_name == "Hotelling T2"
