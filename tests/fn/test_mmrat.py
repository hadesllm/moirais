"""Tests for morie.fn.mmrat -- Maternal mortality ratio."""

import pytest
from morie.fn.mmrat import maternal_mortality_ratio


class TestMaternalMortality:
    def test_known(self):
        res = maternal_mortality_ratio(maternal_deaths=10, live_births=100000)
        assert res.measure == "MMR"
        assert res.estimate == pytest.approx(10.0)

    def test_ci(self):
        res = maternal_mortality_ratio(maternal_deaths=10, live_births=100000)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            maternal_mortality_ratio(maternal_deaths=10, live_births=0)
