"""Tests for moirais.fn.irr -- incidence rate ratio."""

import pytest
from moirais.fn.irr import rate_ratio


class TestRateRatio:
    def test_equal_rates(self):
        """Equal rates should give IRR near 1."""
        result = rate_ratio(
            events1=50, person_time1=1000,
            events2=50, person_time2=1000,
        )
        assert result.estimate == pytest.approx(1.0, rel=0.01)

    def test_double_rate(self):
        """Rate ratio of 100/1000 vs 50/1000 = 2.0."""
        result = rate_ratio(
            events1=100, person_time1=1000,
            events2=50, person_time2=1000,
        )
        assert result.estimate == pytest.approx(2.0, rel=0.01)

    def test_has_se_and_ci(self):
        result = rate_ratio(80, 1000, 40, 1000)
        assert result.se is not None
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
