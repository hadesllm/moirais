"""Tests for morie.fn.wilcox -- Wilcoxon signed-rank test."""

import pytest
from morie.fn.wilcox import wilcoxon_signed_rank_test


class TestWilcoxonSignedRank:
    def test_shifted_pairs(self):
        """Consistently shifted pairs should give small p-value."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x2 = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        result = wilcoxon_signed_rank_test(x1, x2)
        assert isinstance(result, dict)
        assert "statistic" in result
        assert result["p_value"] < 0.05

    def test_identical_pairs(self):
        """Identical pairs should give large p-value."""
        x = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # Add tiny jitter so differences are not exactly zero
        x2 = [5.001, 5.999, 7.001, 7.999, 9.001, 9.999]
        result = wilcoxon_signed_rank_test(x, x2)
        assert result["p_value"] > 0.05

    def test_raises_on_length_mismatch(self):
        """Unequal lengths should raise ValueError."""
        with pytest.raises(ValueError):
            wilcoxon_signed_rank_test([1, 2, 3], [4, 5])
