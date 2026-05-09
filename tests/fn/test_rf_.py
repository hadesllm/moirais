"""Tests for moirais.fn.rf_ — random forest."""
import numpy as np
from moirais.fn.rf_ import random_forest_simple


class TestRandomForest:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 3))
        y = X @ [1, 2, 3] + rng.standard_normal(80) * 0.5
        res = random_forest_simple(X, y, n_trees=10, max_depth=3)
        assert res.value > 0

    def test_n_trees(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        y = rng.standard_normal(50)
        res = random_forest_simple(X, y, n_trees=5)
        assert res.extra["n_trees"] == 5
