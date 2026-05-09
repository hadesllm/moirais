"""Tests for moirais.fn.tpair -- Paired samples t-test."""

import pytest
from moirais.fn.tpair import paired_t_test


class TestPairedTTest:
    def test_shifted_pairs(self):
        """Consistently shifted pairs should reject H0."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [11.0, 12.0, 13.0, 14.0, 15.0]
        result = paired_t_test(x1, x2)
        assert isinstance(result, dict)
        assert "t" in result
        assert result["p_value"] < 0.001

    def test_identical_pairs(self):
        """Identical data should not reject H0."""
        x = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = paired_t_test(x, x)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_raises_on_length_mismatch(self):
        """Unequal lengths should raise ValueError."""
        with pytest.raises(ValueError):
            paired_t_test([1, 2, 3], [4, 5])
