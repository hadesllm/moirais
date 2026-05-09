"""Tests for moirais.fn.getis — Getis-Ord Gi* hot spot statistic."""

import numpy as np
import pytest

from moirais.fn.getis import getis_ord_gi_star


class TestGetisOrdGiStar:

    def test_hot_cluster_positive_z(self):
        """Hot cluster of high values yields positive Gi* z-scores."""
        values = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        W = np.array([
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype=float)
        result = getis_ord_gi_star(values, W)
        assert result.local_values is not None
        assert result.local_values[0] > 0, "Hot spot should have positive z"
        assert result.local_values[5] < 0, "Cold spot should have negative z"

    def test_output_length(self):
        """local_values has same length as input."""
        n = 8
        values = np.arange(n, dtype=float)
        W = np.eye(n)
        result = getis_ord_gi_star(values, W)
        assert len(result.local_values) == n

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            getis_ord_gi_star(np.array([1, 2, 3]), np.eye(4))
