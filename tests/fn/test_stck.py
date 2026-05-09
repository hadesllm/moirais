"""Tests for moirais.fn.stck -- Model stacking."""

import numpy as np
from moirais.fn.stck import stacking, stck
from moirais.fn._containers import DescriptiveResult


class TestStck:
    def test_alias(self):
        assert stck is stacking

    def test_two_models(self):
        rng = np.random.default_rng(42)
        n = 50
        X = rng.normal(0, 1, (n, 2))
        y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, n)
        pred1 = X[:, 0] * 0.9
        pred2 = (X[:, 0] + X[:, 1]) * 0.6
        result = stacking(X, y, [pred1, pred2])
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_base"] == 2
        assert len(result.extra["predictions"]) == n

    def test_r_squared_reasonable(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 30)
        y = 2 * x + 1
        result = stacking(x, y, [2 * x, x + 1])
        assert result.extra["r_squared"] > 0.9
