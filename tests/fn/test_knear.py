"""Tests for moirais.fn.knear — K-nearest neighbors spatial weights."""

import numpy as np
import pytest

from moirais.fn.knear import knn_weights


class TestKNNWeights:

    def test_each_row_has_k_neighbors(self):
        """Each row of the weight matrix has exactly k nonzero entries."""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        result = knn_weights(coords, k=2)
        W = result.value
        for i in range(5):
            assert W[i].sum() == pytest.approx(2.0)

    def test_diagonal_is_zero(self):
        """No self-neighbors (diagonal is zero)."""
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        result = knn_weights(coords, k=2)
        W = result.value
        assert np.all(np.diag(W) == 0)

    def test_k_too_large_raises(self):
        """k >= n raises ValueError."""
        coords = np.array([[0, 0], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="k.*must be < n"):
            knn_weights(coords, k=2)
