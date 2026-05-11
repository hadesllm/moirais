"""Tests for box_m_test."""
import numpy as np, pytest
from morie.fn.box_m import box_m_test

class TestBoxM:
    def test_equal_cov(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 2))
        g = np.array([0]*20 + [1]*20)
        r = box_m_test(X, g)
        assert r.p_value > 0.01

    def test_returns_fields(self):
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(0, 3, (20, 2))])
        g = np.array([0]*20 + [1]*20)
        r = box_m_test(X, g)
        assert r.test_name == "Box M"
