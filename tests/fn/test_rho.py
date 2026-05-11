"""Tests for morie.fn.rho -- Spearman rank correlation."""

import numpy as np
import pytest
from morie.fn.rho import spearman_rho


class TestSpearmanRho:
    def test_perfect_rank_correlation(self):
        """Monotonically increasing data gives rho = 1."""
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [10, 20, 30, 40, 50, 60, 70, 80]
        result = spearman_rho(x, y)
        assert isinstance(result, dict)
        assert result["rho"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        """Monotonically decreasing gives rho = -1."""
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        result = spearman_rho(x, y)
        assert result["rho"] == pytest.approx(-1.0, abs=1e-10)

    def test_p_value_significant(self):
        """Strong correlation should produce small p-value."""
        x = list(range(20))
        y = list(range(20))
        result = spearman_rho(x, y)
        assert result["p_value"] < 0.001
