"""Tests for moirais.fn.rbc -- Rank-biserial correlation."""

import pytest
from moirais.fn.rbc import rank_biserial_correlation
from moirais.fn._containers import ESRes


class TestRankBiserialCorrelation:
    def test_completely_separated(self):
        """Complete dominance of x over y gives r near 1 or -1."""
        x = [10.0, 11.0, 12.0, 13.0]
        y = [1.0, 2.0, 3.0, 4.0]
        result = rank_biserial_correlation(x, y)
        assert isinstance(result, ESRes)
        assert abs(result.estimate) > 0.8

    def test_identical_groups_near_zero(self):
        """Same values in both groups give r near 0."""
        x = [1.0, 2.0, 3.0, 4.0]
        result = rank_biserial_correlation(x, x)
        assert result.estimate == pytest.approx(0.0, abs=0.1)

    def test_bounded(self):
        """Rank-biserial r should be in [-1, 1]."""
        x = [1.0, 5.0, 9.0, 13.0]
        y = [2.0, 6.0, 10.0, 14.0]
        result = rank_biserial_correlation(x, y)
        assert -1.0 <= result.estimate <= 1.0
