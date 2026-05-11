"""Tests for morie.fn.nn_ -- Simple neural network."""

import numpy as np
import pytest
from morie.fn.nn_ import nn_classify


class TestNnClassify:
    def test_xor_like(self):
        """Should learn XOR-like pattern with a hidden layer."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
        y = np.array([0, 1, 1, 0] * 10, dtype=float)
        result = nn_classify(X, y, hidden_size=16, epochs=500, lr=0.5)
        assert "predictions" in result
        assert "loss_history" in result
        assert len(result["loss_history"]) == 500
        # Loss should decrease
        assert result["loss_history"][-1] < result["loss_history"][0]

    def test_separable(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [3, 3],
                        rng.standard_normal((30, 2)) - [3, 3]])
        y = np.array([1.0] * 30 + [0.0] * 30)
        result = nn_classify(X, y, hidden_size=8, epochs=200, lr=0.1)
        acc = np.mean(result["predictions"] == y)
        assert acc >= 0.7

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            nn_classify(np.zeros((5, 2)), np.zeros(3))
