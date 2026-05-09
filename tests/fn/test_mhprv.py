"""Tests for moirais.fn.mhprv -- mental health prevalence."""

import pytest
from moirais.fn.mhprv import mental_health_prevalence


class TestMentalHealthPrevalence:
    def test_basic(self):
        res = mental_health_prevalence(n_cases=50, n_surveyed=500)
        assert res.estimate == pytest.approx(0.1)

    def test_ci(self):
        res = mental_health_prevalence(50, 500)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            mental_health_prevalence(10, 0)
