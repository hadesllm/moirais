"""Tests for morie.fn.cart — CART decision tree."""
import numpy as np
import pytest
from morie.fn.cart import decision_tree


class TestCART:
    def test_perfect_split(self):
        X = np.array([[0], [0], [1], [1]], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=float)
        res = decision_tree(X, y, max_depth=2)
        assert res.value == pytest.approx(1.0, abs=0.01)

    def test_r2_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ [1, 2, 3] + rng.standard_normal(100) * 0.5
        res = decision_tree(X, y, max_depth=5)
        assert res.value > 0.5
