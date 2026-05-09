"""Tests for moirais.fn.t2smp -- Two-sample t-test."""

import pytest
from moirais.fn.t2smp import two_sample_t_test


class TestTwoSampleTTest:
    def test_different_means(self):
        """Well-separated samples should reject H0."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = two_sample_t_test(x1, x2)
        assert isinstance(result, dict)
        assert "t" in result
        assert result["p_value"] < 0.001

    def test_same_distribution(self):
        """Identical samples should not reject H0."""
        x = [5.0, 5.1, 4.9, 5.2, 4.8]
        result = two_sample_t_test(x, x)
        assert result["p_value"] > 0.05

    def test_welch_is_default(self):
        """Default method should be Welch's t-test."""
        result = two_sample_t_test([1, 2, 3], [4, 5, 6])
        assert "method" in result
        assert "Welch" in result["method"] or "welch" in result["method"].lower()
