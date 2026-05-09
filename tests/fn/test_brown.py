"""Tests for brown_forsythe."""
import numpy as np, pytest
from moirais.fn.brown import brown_forsythe

class TestBrown:
    def test_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        r = brown_forsythe(a, b)
        assert r.p_value > 0.05

    def test_unequal(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 4, 50)
        r = brown_forsythe(a, b)
        assert r.test_name == "Brown-Forsythe"
