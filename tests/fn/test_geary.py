"""Tests for moirais.fn.geary — Geary's C spatial autocorrelation."""

import numpy as np
import pytest

from moirais.fn.geary import gearys_c


class TestGearysC:

    def test_clustered_C_less_than_one(self):
        """Clustered data produces C < 1."""
        values = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
        W = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float)
        result = gearys_c(values, W, nperm=199, seed=42)
        assert result.statistic < 1.0, f"Expected C < 1, got {result.statistic}"

    def test_expected_is_one(self):
        """Expected value under null is 1.0."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        result = gearys_c(values, W, nperm=99, seed=42)
        assert result.expected == 1.0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            gearys_c(np.array([1, 2]), np.eye(3))
