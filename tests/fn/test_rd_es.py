"""Tests for morie.fn.rd_es -- risk difference effect size."""

import pytest
from morie.fn.rd_es import risk_difference


class TestRiskDifference:
    def test_known_values(self):
        """a=20,b=80,c=10,d=90: p1=0.2, p2=0.1, RD=0.1."""
        result = risk_difference(a=20, b=80, c=10, d=90)
        assert result.measure == "Risk difference"
        assert result.estimate == pytest.approx(0.1, abs=0.001)

    def test_equal_proportions(self):
        """Equal proportions give RD=0."""
        result = risk_difference(a=30, b=70, c=30, d=70)
        assert result.estimate == pytest.approx(0.0, abs=1e-10)

    def test_has_se_and_n(self):
        result = risk_difference(a=25, b=75, c=15, d=85)
        assert result.se is not None
        assert result.se > 0
        assert result.n == 200
