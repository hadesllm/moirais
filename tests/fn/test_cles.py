"""Tests for moirais.fn.cles -- Common Language Effect Size."""

import pytest
from moirais.fn.cles import cles
from moirais.fn._containers import ESRes


class TestCLES:
    def test_well_separated_near_one(self):
        """Completely separated groups give CLES near 1."""
        x = [10.0, 11.0, 12.0, 13.0, 14.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = cles(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.95

    def test_identical_groups_near_half(self):
        """Identical distributions give CLES near 0.5."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = cles(x, y)
        assert result.estimate == pytest.approx(0.5, abs=0.15)

    def test_bounded_zero_one(self):
        """CLES should be in [0, 1]."""
        x = [1.0, 3.0, 5.0, 7.0]
        y = [2.0, 4.0, 6.0, 8.0]
        result = cles(x, y)
        assert 0.0 <= result.estimate <= 1.0
