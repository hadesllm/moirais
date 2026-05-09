"""Tests for moirais.fn.rr_ci -- incidence rate ratio with CI."""

import pytest
from moirais.fn.rr_ci import rate_ratio_ci


class TestRateRatioCI:
    def test_equal_rates(self):
        """Equal rates should give IRR near 1."""
        result = rate_ratio_ci(n1=50, t1=1000, n2=50, t2=1000)
        assert result["irr"] == pytest.approx(1.0, rel=0.01)
        assert result["ci_lower"] < 1.0
        assert result["ci_upper"] > 1.0

    def test_double_rate(self):
        """Rate 100/1000 vs 50/1000 should give IRR=2."""
        result = rate_ratio_ci(n1=100, t1=1000, n2=50, t2=1000)
        assert result["irr"] == pytest.approx(2.0, rel=0.01)
        assert result["ci_lower"] > 1.0

    def test_invalid_person_time_raises(self):
        with pytest.raises(ValueError, match="Person-time"):
            rate_ratio_ci(n1=10, t1=0, n2=10, t2=100)
