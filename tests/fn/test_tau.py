"""Tests for morie.fn.tau -- Kendall's tau-b rank correlation."""

import pytest
from morie.fn.tau import kendall_tau


class TestKendallTau:
    def test_perfect_concordance(self):
        """Perfectly concordant ranks give tau = 1."""
        x = [1, 2, 3, 4, 5, 6]
        y = [10, 20, 30, 40, 50, 60]
        result = kendall_tau(x, y)
        assert isinstance(result, dict)
        assert result["tau"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_discordance(self):
        """Reversed ranks give tau = -1."""
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        result = kendall_tau(x, y)
        assert result["tau"] == pytest.approx(-1.0, abs=1e-10)

    def test_returns_p_value(self):
        """Result dict should contain p_value in [0, 1]."""
        result = kendall_tau([1, 2, 3, 4], [2, 4, 6, 8])
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
