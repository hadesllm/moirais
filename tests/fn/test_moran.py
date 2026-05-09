"""Tests for moirais.fn.moran — Moran's I global spatial autocorrelation."""

import numpy as np
import pytest

from moirais.fn.moran import morans_i


class TestMoransI:

    def test_clustered_positive_I(self):
        """Clustered values produce positive Moran's I."""
        values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        W = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float)
        result = morans_i(values, W, nperm=499, seed=42)
        assert result.statistic > 0, f"Expected positive I, got {result.statistic}"
        assert result.name == "Moran's I"

    def test_random_values_near_expected(self):
        """Random values should produce I near E[I] = -1/(n-1)."""
        rng = np.random.default_rng(123)
        n = 20
        values = rng.standard_normal(n)
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        result = morans_i(values, W, nperm=199, seed=42)
        expected = -1.0 / (n - 1)
        assert abs(result.statistic - expected) < 1.0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="weights must be"):
            morans_i(np.array([1, 2, 3]), np.eye(4))
