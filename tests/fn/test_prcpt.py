"""Tests for morie.fn.prcpt — perceptron."""
import numpy as np
from morie.fn.prcpt import perceptron


class TestPerceptron:
    def test_linearly_separable(self):
        X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]], dtype=float)
        y = np.array([1, 1, -1, -1], dtype=float)
        res = perceptron(X, y, n_iter=100, lr=0.1)
        assert res.value >= 0.75

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = np.sign(X @ [1, 0, 0])
        y[y == 0] = 1
        res = perceptron(X, y)
        assert len(res.extra["predictions"]) == 20
