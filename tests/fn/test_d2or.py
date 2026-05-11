"""Tests for morie.fn.d2or -- Convert Cohen's d to odds ratio."""

import math
import pytest
from morie.fn.d2or import d_to_or


class TestDToOR:
    def test_zero_d_gives_one(self):
        """d=0 should give OR=1 (no effect)."""
        assert d_to_or(0.0) == pytest.approx(1.0, abs=1e-10)

    def test_positive_d_greater_than_one(self):
        """Positive d gives OR > 1."""
        assert d_to_or(0.8) > 1.0

    def test_known_formula(self):
        """OR = exp(d * pi / sqrt(3))."""
        d = 0.5
        expected = math.exp(d * math.pi / math.sqrt(3))
        assert d_to_or(d) == pytest.approx(expected, abs=1e-10)
