"""Tests for morie.fn.vacnnt -- vaccine NNV."""

import pytest
from morie.fn.vacnnt import vaccine_nnt


class TestVaccineNNT:
    def test_basic(self):
        res = vaccine_nnt(vaccine_efficacy=0.8, baseline_risk=0.05)
        assert res.estimate == pytest.approx(25.0)

    def test_perfect_efficacy(self):
        res = vaccine_nnt(1.0, 0.1)
        assert res.estimate == pytest.approx(10.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            vaccine_nnt(0, 0.05)
