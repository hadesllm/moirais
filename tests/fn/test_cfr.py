"""Tests for morie.fn.cfr -- case fatality rate."""

import pytest
from morie.fn.cfr import case_fatality_rate


class TestCFR:
    def test_known(self):
        """10 deaths / 100 cases => CFR = 0.1."""
        res = case_fatality_rate(deaths=10, cases=100)
        assert res.measure == "CFR"
        assert res.estimate == pytest.approx(0.1)

    def test_ci_bounds(self):
        """CI should bracket the estimate."""
        res = case_fatality_rate(deaths=10, cases=100)
        assert res.ci_lower < res.estimate
        assert res.ci_upper > res.estimate

    def test_zero_cases_raises(self):
        """Zero cases should raise."""
        with pytest.raises(ValueError):
            case_fatality_rate(deaths=0, cases=0)
