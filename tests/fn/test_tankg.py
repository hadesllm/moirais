"""Tests for moirais.fn.tankg -- neural network training."""

import numpy as np
from moirais.fn.tankg import nn_train, tankg
from moirais.fn._containers import DescriptiveResult


class TestTankg:
    def test_alias(self):
        assert tankg is nn_train

    def test_xor_like(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
        y = np.array([0, 1, 1, 0] * 25, dtype=float)
        result = nn_train(X, y, hidden=8, epochs=500, lr=0.5, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 0.5

    def test_linearly_separable(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        y = (X[:, 0] > 0).astype(float)
        result = nn_train(X, y, hidden=16, epochs=500, lr=0.1, seed=42)
        assert result.value >= 0.6
