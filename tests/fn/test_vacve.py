"""Tests for moirais.fn.vacve -- vaccine effectiveness."""

import pytest
from moirais.fn.vacve import vaccine_effectiveness


class TestVaccineEffectiveness:
    def test_basic(self):
        res = vaccine_effectiveness(
            cases_vacc=10, cases_unvacc=50,
            pop_vacc=5000, pop_unvacc=5000,
        )
        assert res.estimate == pytest.approx(80.0)

    def test_ci(self):
        res = vaccine_effectiveness(10, 50, 5000, 5000)
        assert res.ci_lower < res.estimate
