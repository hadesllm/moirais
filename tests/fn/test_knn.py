"""Tests for morie.fn.knn -- k-Nearest Neighbors."""

import numpy as np
import pytest
from morie.fn.knn import knn_classify


class TestKnnClassify:
    def test_perfect_classification(self):
        """Identical train/test on well-separated clusters."""
        X_train = np.array([[0, 0], [0, 1], [1, 0],
                            [10, 10], [10, 11], [11, 10]], dtype=float)
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[0.5, 0.5], [10.5, 10.5]], dtype=float)
        result = knn_classify(X_train, y_train, X_test, k=3)
        assert list(result["predictions"]) == [0, 1]

    def test_distances_shape(self):
        rng = np.random.default_rng(42)
        X_tr = rng.standard_normal((20, 3))
        y_tr = np.ones(20)
        X_te = rng.standard_normal((5, 3))
        result = knn_classify(X_tr, y_tr, X_te, k=3)
        assert result["distances"].shape == (5, 3)

    def test_k_too_large_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            knn_classify(np.zeros((2, 1)), np.array([0, 1]),
                         np.zeros((1, 1)), k=5)
