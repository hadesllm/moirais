"""Tests for moirais.fn.t1smp -- One-sample t-test."""

import pytest
from moirais.fn.t1smp import one_sample_t_test


class TestOneSampleTTest:
    def test_known_mean(self):
        """Sample centered at 5 tested against mu0=0 should reject."""
        x = [4.5, 5.0, 5.5, 5.2, 4.8, 5.1, 5.3, 4.9]
        result = one_sample_t_test(x, mu0=0.0)
        assert isinstance(result, dict)
        assert "t" in result
        assert result["p_value"] < 0.001

    def test_correct_mu_not_significant(self):
        """Sample centered at 5 tested against mu0=5 should not reject."""
        x = [4.5, 5.0, 5.5, 5.2, 4.8, 5.1, 5.3, 4.9]
        result = one_sample_t_test(x, mu0=5.0)
        assert result["p_value"] > 0.05

    def test_returns_ci(self):
        """Result should contain CI bounds."""
        result = one_sample_t_test([1, 2, 3, 4, 5])
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["ci_upper"]
