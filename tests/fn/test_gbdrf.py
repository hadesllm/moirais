"""Tests for morie.fn.gbdrf -- GBD risk factor."""

import pytest
from morie.fn.gbdrf import gbd_risk_factor


class TestGBDRiskFactor:
    def test_basic(self):
        res = gbd_risk_factor(rr=2.0, prevalence=0.3)
        expected_paf = 0.3 * 1.0 / (0.3 * 1.0 + 1)
        assert res.value == pytest.approx(expected_paf)

    def test_zero_prev(self):
        res = gbd_risk_factor(rr=2.0, prevalence=0.0)
        assert res.value == pytest.approx(0.0)
