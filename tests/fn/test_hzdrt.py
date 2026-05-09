"""Tests for moirais.fn.hzdrt -- Hazard rate from life table."""

import pytest
from moirais.fn.hzdrt import hazard_rate


class TestHazardRate:
    def test_known(self):
        res = hazard_rate(deaths=[10, 20], person_years=[1000.0, 2000.0])
        assert res.estimate == pytest.approx(30 / 3000)

    def test_ci(self):
        res = hazard_rate(deaths=[50], person_years=[10000.0])
        assert res.extra["ci_lower"][0] < res.extra["rates"][0]
        assert res.extra["ci_upper"][0] > res.extra["rates"][0]

    def test_invalid(self):
        with pytest.raises(ValueError):
            hazard_rate(deaths=[10], person_years=[0.0])
