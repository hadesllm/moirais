"""Tests for moirais.fn.lisa — Local Moran's I (LISA)."""

import numpy as np
import pytest

from moirais.fn.lisa import local_morans_i


class TestLISA:

    def test_local_values_length(self):
        """local_values array has same length as input."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        W = np.zeros((6, 6))
        for i in range(5):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        result = local_morans_i(values, W, nperm=99, seed=42)
        assert result.local_values is not None
        assert len(result.local_values) == len(values)

    def test_clustered_global_positive(self):
        """Clustered values yield positive global statistic."""
        values = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
        W = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float)
        result = local_morans_i(values, W, nperm=199, seed=42)
        assert result.statistic > 0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            local_morans_i(np.array([1, 2, 3]), np.eye(4))
