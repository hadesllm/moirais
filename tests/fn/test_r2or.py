"""Tests for moirais.fn.r2or -- Convert Pearson r to odds ratio."""

import pytest
from moirais.fn.r2or import r_to_or


class TestRToOR:
    def test_zero_r_gives_or_one(self):
        """r=0 should give OR=1."""
        assert r_to_or(0.0) == pytest.approx(1.0, abs=1e-10)

    def test_positive_r_or_greater_one(self):
        """Positive r gives OR > 1."""
        assert r_to_or(0.3) > 1.0

    def test_negative_r_or_less_one(self):
        """Negative r gives OR < 1."""
        assert r_to_or(-0.3) < 1.0
