"""Tests for moirais.fn.dalys -- DALYs."""

import pytest
from moirais.fn.dalys import disability_adjusted_life_years, yld_from_prevalence


class TestDALY:
    def test_basic(self):
        res = disability_adjusted_life_years(100.0, 50.0)
        assert res["daly"] == pytest.approx(150.0)

    def test_pct_yll(self):
        res = disability_adjusted_life_years(75.0, 25.0)
        assert res["pct_yll"] == pytest.approx(75.0)
        assert res["pct_yld"] == pytest.approx(25.0)

    def test_zero_daly(self):
        res = disability_adjusted_life_years(0.0, 0.0)
        assert res["daly"] == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            disability_adjusted_life_years(-10.0, 5.0)


class TestYLD:
    def test_basic(self):
        res = yld_from_prevalence(1000.0, 0.5, duration=1.0, discount_rate=0.0)
        assert res["yld"] == pytest.approx(500.0)

    def test_discounted_less(self):
        r0 = yld_from_prevalence(100.0, 0.3, duration=5.0, discount_rate=0.0)
        r3 = yld_from_prevalence(100.0, 0.3, duration=5.0, discount_rate=0.03)
        assert r3["yld_discounted"] < r0["yld"]

    def test_invalid_dw_raises(self):
        with pytest.raises(ValueError):
            yld_from_prevalence(100.0, 1.5)
