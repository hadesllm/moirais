"""Tests for morie.fn.or_es -- odds ratio effect size."""

import pytest
from morie.fn.or_es import odds_ratio


class TestOddsRatio:
    def test_known_2x2(self):
        """OR for a=10,b=5,c=3,d=12: (10*12)/(5*3) = 8.0."""
        result = odds_ratio(a=10, b=5, c=3, d=12)
        assert result.measure == "Odds ratio"
        assert result.estimate == pytest.approx(8.0, rel=0.01)

    def test_null_effect(self):
        """Equal cells should give OR near 1."""
        result = odds_ratio(a=50, b=50, c=50, d=50)
        assert result.estimate == pytest.approx(1.0, rel=0.01)

    def test_has_ci_and_extra(self):
        result = odds_ratio(a=20, b=30, c=15, d=35)
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
        assert "log_or" in result.extra
