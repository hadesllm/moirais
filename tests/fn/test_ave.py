"""Tests for morie.fn.ave — Average Variance Extracted."""

import pytest
import numpy as np
from morie.fn import ave


class TestAve:
    """Tests for AVE from factor loadings."""

    def test_known_loadings(self):
        """Known loadings [0.7, 0.8, 0.9] -> AVE = mean of squares.

        AVE = (0.49 + 0.64 + 0.81) / 3 = 0.6467
        """
        loads = np.array([0.7, 0.8, 0.9])
        result = ave(loads)
        expected = (0.49 + 0.64 + 0.81) / 3
        assert result == pytest.approx(expected, abs=1e-4)

    def test_perfect_loadings(self):
        """All 1.0 loadings should yield AVE = 1.0."""
        assert ave(np.array([1.0, 1.0])) == pytest.approx(1.0)

    def test_zero_loadings(self):
        """All-zero loadings should yield AVE = 0."""
        assert ave(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_single_loading(self):
        """Single loading: AVE = loading^2."""
        assert ave(np.array([0.6])) == pytest.approx(0.36, abs=1e-10)

    def test_convergent_validity_threshold(self):
        """High loadings should pass the Fornell-Larcker threshold of 0.5."""
        loads = np.array([0.8, 0.85, 0.9])
        assert ave(loads) >= 0.5
