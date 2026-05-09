"""Tests for moirais.fn.ird -- incidence rate difference."""

import pytest
from moirais.fn.ird import incidence_rate_difference


class TestIncidenceRateDifference:
    def test_known_values(self):
        """50 events in 1000 PY vs 30 events in 1000 PY: IRD = 0.02."""
        result = incidence_rate_difference(
            events1=50, person_time1=1000,
            events2=30, person_time2=1000,
        )
        assert result.measure == "Incidence rate difference"
        assert result.estimate == pytest.approx(0.02, abs=0.001)

    def test_equal_rates_zero(self):
        """Equal rates should give IRD near 0."""
        result = incidence_rate_difference(
            events1=40, person_time1=1000,
            events2=40, person_time2=1000,
        )
        assert result.estimate == pytest.approx(0.0, abs=1e-10)

    def test_has_ci(self):
        result = incidence_rate_difference(50, 1000, 30, 1000)
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
