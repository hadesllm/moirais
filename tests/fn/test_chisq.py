"""Tests for moirais.fn.chisq -- Chi-square test."""

import numpy as np
import pytest
from moirais.fn.chisq import chi_square_test


class TestChiSquareTest:
    def test_2d_independence(self):
        """2x2 contingency table with clear association."""
        table = [[90, 10], [10, 90]]
        result = chi_square_test(table)
        assert isinstance(result, dict)
        assert "chi2" in result
        assert result["chi2"] > 50
        assert result["p_value"] < 0.001

    def test_1d_goodness_of_fit(self):
        """Uniform observed matches uniform expected, chi2 near 0."""
        result = chi_square_test([25, 25, 25, 25])
        assert result["chi2"] == pytest.approx(0.0, abs=1e-10)
        assert result["p_value"] > 0.99

    def test_raises_on_negative(self):
        """Negative frequencies should raise ValueError."""
        with pytest.raises(ValueError):
            chi_square_test([-5, 10])
