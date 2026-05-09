"""Tests for moirais.fn.mlp — simple MLP."""
import numpy as np
from moirais.fn.mlp import mlp_simple


class TestMLP:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 3))
        y = X @ [1, 2, 0] + rng.standard_normal(80) * 0.1
        res = mlp_simple(X, y, hidden=16, n_iter=200, lr=0.005)
        assert res.value > 0.3

    def test_loss_decreases(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        y = X @ [1, -1]
        res = mlp_simple(X, y, hidden=8, n_iter=100)
        assert res.extra["final_loss"] < np.var(y)
