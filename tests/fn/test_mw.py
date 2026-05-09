"""Tests for moirais.fn.mw -- Mann-Whitney U test."""

import pytest
from moirais.fn.mw import mann_whitney_test


class TestMannWhitneyTest:
    def test_different_groups(self):
        """Well-separated groups should give small p-value."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = mann_whitney_test(x1, x2)
        assert isinstance(result, dict)
        assert "U" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05

    def test_identical_groups_not_significant(self):
        """Same data should yield large p-value."""
        x = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = mann_whitney_test(x, x)
        assert result["p_value"] > 0.05

    def test_raises_on_empty(self):
        """Empty sample should raise ValueError."""
        with pytest.raises(ValueError):
            mann_whitney_test([], [1, 2, 3])
