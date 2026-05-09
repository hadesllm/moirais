"""Tests for moirais.fn.cohrt -- Cohort risk ratio."""

import pytest
from moirais.fn.cohrt import cohort_risk_ratio


class TestCohortRR:
    def test_known(self):
        res = cohort_risk_ratio(a=30, b=70, c=10, d=90)
        assert res.measure == "RR_cohort"
        assert res.estimate == pytest.approx(3.0)

    def test_ci(self):
        res = cohort_risk_ratio(a=30, b=70, c=10, d=90)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_zero_unexposed(self):
        with pytest.raises(ValueError):
            cohort_risk_ratio(a=10, b=90, c=0, d=100)
