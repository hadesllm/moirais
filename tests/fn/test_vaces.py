"""Tests for moirais.fn.vaces -- vaccine efficacy (Greenwood/exact)."""

import numpy as np
import pytest
from moirais.fn.vaces import vaccine_efficacy_exact


class TestVaccineEfficacyExact:
    def test_high_efficacy(self):
        res = vaccine_efficacy_exact(5, 1000, 50, 1000)
        assert res["ve"] == pytest.approx((1 - 5 / 50) * 100, abs=0.1)

    def test_relative_risk(self):
        res = vaccine_efficacy_exact(10, 500, 40, 500)
        assert res["relative_risk"] == pytest.approx(0.25)

    def test_greenwood_ci_ordering(self):
        res = vaccine_efficacy_exact(8, 1000, 40, 1000)
        assert res["ci_lower_greenwood"] < res["ve"] < res["ci_upper_greenwood"]

    def test_zero_vacc_cases(self):
        res = vaccine_efficacy_exact(0, 1000, 50, 1000)
        assert res["ve"] == pytest.approx(100.0)

    def test_negative_group_raises(self):
        with pytest.raises(ValueError):
            vaccine_efficacy_exact(5, -1, 10, 100)
