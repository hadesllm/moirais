"""Tests for morie.fn.cliff -- Cliff's delta (non-parametric effect size)."""

import pytest
from morie.fn.cliff import cliffs_delta
from morie.fn._containers import ESRes


class TestCliffsDelta:
    def test_completely_separated(self):
        """All x > y gives delta = 1."""
        x = [10.0, 11.0, 12.0]
        y = [1.0, 2.0, 3.0]
        result = cliffs_delta(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(1.0, abs=1e-10)

    def test_identical_groups_zero(self):
        """Identical groups give delta = 0."""
        x = [1.0, 2.0, 3.0]
        result = cliffs_delta(x, x)
        assert result.estimate == pytest.approx(0.0, abs=0.01)

    def test_bounded(self):
        """Cliff's delta should be in [-1, 1]."""
        x = [1.0, 5.0, 9.0]
        y = [2.0, 6.0, 10.0]
        result = cliffs_delta(x, y)
        assert -1.0 <= result.estimate <= 1.0
