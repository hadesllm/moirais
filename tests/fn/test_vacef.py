"""Tests for moirais.fn.vacef -- vaccine efficacy."""

import pytest
from moirais.fn.vacef import vaccine_efficacy


class TestVaccineEfficacy:
    def test_basic(self):
        res = vaccine_efficacy(ar_vaccinated=0.01, ar_placebo=0.05)
        assert res.estimate == pytest.approx(80.0)

    def test_with_n(self):
        res = vaccine_efficacy(0.01, 0.05, n_vaccinated=10000, n_placebo=10000)
        assert res.ci_lower is not None
        assert res.ci_lower < res.estimate < res.ci_upper
