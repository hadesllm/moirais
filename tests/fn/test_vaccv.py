"""Tests for morie.fn.vaccv -- vaccine coverage."""

import pytest
from morie.fn.vaccv import vaccine_coverage


class TestVaccineCoverage:
    def test_basic(self):
        res = vaccine_coverage(n_vaccinated=800, n_eligible=1000)
        assert res.estimate == pytest.approx(0.8)

    def test_ci(self):
        res = vaccine_coverage(800, 1000)
        assert res.ci_lower < res.estimate < res.ci_upper
