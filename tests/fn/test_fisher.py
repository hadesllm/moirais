"""Tests for moirais.fn.fisher -- Fisher's exact test."""

import pytest
from moirais.fn.fisher import fisher_exact_test


class TestFisherExact:
    def test_known_table(self):
        """Strong association should yield small p-value."""
        table = [[10, 0], [0, 10]]
        result = fisher_exact_test(table)
        assert isinstance(result, dict)
        assert "odds_ratio" in result
        assert "p_value" in result
        assert result["p_value"] < 0.01

    def test_independent_table(self):
        """Balanced table gives OR near 1, large p."""
        table = [[5, 5], [5, 5]]
        result = fisher_exact_test(table)
        assert result["odds_ratio"] == pytest.approx(1.0, abs=0.01)
        assert result["p_value"] > 0.5

    def test_raises_on_wrong_shape(self):
        """Non-2x2 table should raise ValueError."""
        with pytest.raises(ValueError):
            fisher_exact_test([[1, 2, 3], [4, 5, 6]])
