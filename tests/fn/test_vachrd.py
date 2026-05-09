"""Tests for moirais.fn.vachrd -- vaccine herd immunity."""

import pytest
from moirais.fn.vachrd import vaccine_herd


class TestVaccineHerd:
    def test_measles(self):
        res = vaccine_herd(R0=15, vaccine_efficacy=0.95)
        assert res.estimate > 0.9

    def test_low_r0(self):
        res = vaccine_herd(R0=0.8)
        assert res.estimate == 0.0

    def test_perfect_efficacy(self):
        res = vaccine_herd(R0=3.0, vaccine_efficacy=1.0)
        expected = 1 - 1 / 3.0
        assert res.estimate == pytest.approx(expected)
