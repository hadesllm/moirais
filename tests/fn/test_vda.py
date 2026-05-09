"""Tests for moirais.fn.vda -- Vargha-Delaney A statistic."""

import pytest
from moirais.fn.vda import vargha_delaney_a
from moirais.fn._containers import ESRes


class TestVarghaDelaneyA:
    def test_equal_groups_near_half(self):
        """Identical distributions yield A near 0.5."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = vargha_delaney_a(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(0.5, abs=0.05)

    def test_completely_dominant(self):
        """All x > y gives A near 1."""
        x = [10.0, 11.0, 12.0, 13.0]
        y = [1.0, 2.0, 3.0, 4.0]
        result = vargha_delaney_a(x, y)
        assert result.estimate > 0.9

    def test_bounded_zero_one(self):
        """A should be in [0, 1]."""
        x = [1.0, 3.0, 5.0]
        y = [2.0, 4.0, 6.0]
        result = vargha_delaney_a(x, y)
        assert 0.0 <= result.estimate <= 1.0
