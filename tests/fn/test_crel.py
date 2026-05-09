"""Tests for moirais.fn.crel — composite reliability."""

import pytest
import numpy as np
from moirais.fn import crel


class TestCrel:
    """Tests for composite reliability from factor loadings."""

    def test_known_loadings(self):
        """Known loadings [0.7, 0.8, 0.9] should give a known CR value.

        CR = (0.7+0.8+0.9)^2 / ((0.7+0.8+0.9)^2 + (1-0.49)+(1-0.64)+(1-0.81))
           = 5.76 / (5.76 + 0.51 + 0.36 + 0.19)
           = 5.76 / 6.82
        """
        loads = np.array([0.7, 0.8, 0.9])
        result = crel(loads)
        expected = 5.76 / (5.76 + 0.51 + 0.36 + 0.19)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_all_zeros_returns_zero(self):
        """All-zero loadings should return 0 (0/0 edge — numerator is 0)."""
        loads = np.array([0.0, 0.0, 0.0])
        result = crel(loads)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_perfect_loadings(self):
        """Loadings of 1.0 should yield CR = 1.0."""
        loads = np.array([1.0, 1.0, 1.0])
        result = crel(loads)
        # (3)^2 / (9 + 0) = 1.0
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_single_item(self):
        """Single loading should give valid CR."""
        result = crel(np.array([0.8]))
        # (0.8)^2 / (0.64 + 0.36) = 0.64
        assert result == pytest.approx(0.64, abs=1e-10)
