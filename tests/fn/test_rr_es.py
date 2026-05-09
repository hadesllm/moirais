"""Tests for moirais.fn.rr_es -- risk ratio effect size."""

import pytest
from moirais.fn.rr_es import risk_ratio


class TestRiskRatio:
    def test_known_values(self):
        """a=20,b=80,c=10,d=90: p1=0.2, p2=0.1, RR=2.0."""
        result = risk_ratio(a=20, b=80, c=10, d=90)
        assert result.measure == "Risk ratio"
        assert result.estimate == pytest.approx(2.0, rel=0.01)

    def test_equal_risks(self):
        """Equal risks give RR=1."""
        result = risk_ratio(a=30, b=70, c=30, d=70)
        assert result.estimate == pytest.approx(1.0, rel=0.01)

    def test_ci_contains_point(self):
        result = risk_ratio(a=40, b=60, c=20, d=80)
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
