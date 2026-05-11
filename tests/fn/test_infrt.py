"""Tests for morie.fn.infrt -- Infant mortality rate."""

import pytest
from morie.fn.infrt import infant_mortality_rate


class TestInfantMortality:
    def test_known(self):
        res = infant_mortality_rate(infant_deaths=50, live_births=10000)
        assert res.measure == "IMR"
        assert res.estimate == pytest.approx(5.0)

    def test_ci(self):
        res = infant_mortality_rate(infant_deaths=50, live_births=10000)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            infant_mortality_rate(infant_deaths=10, live_births=0)
