"""Tests for morie.fn.yllis -- years of life lost."""

import numpy as np
import pytest
from morie.fn.yllis import years_of_life_lost


class TestYLL:
    def test_undiscounted(self):
        ages = np.array([30.0, 50.0, 70.0])
        res = years_of_life_lost(ages, life_expectancy=80.0, discount_rate=0.0)
        expected = (80 - 30) + (80 - 50) + (80 - 70)
        assert res["total_yll"] == pytest.approx(expected)

    def test_discounted_less_than_undiscounted(self):
        ages = np.array([40.0, 60.0])
        r0 = years_of_life_lost(ages, discount_rate=0.0)
        r3 = years_of_life_lost(ages, discount_rate=0.03)
        assert r3["total_yll"] < r0["total_yll"]

    def test_age_at_expectancy(self):
        ages = np.array([80.0])
        res = years_of_life_lost(ages, life_expectancy=80.0, discount_rate=0.0)
        assert res["total_yll"] == pytest.approx(0.0)

    def test_per_individual_le(self):
        ages = np.array([30.0, 60.0])
        le = np.array([85.0, 78.0])
        res = years_of_life_lost(ages, life_expectancy=le, discount_rate=0.0)
        assert res["total_yll"] == pytest.approx(55.0 + 18.0)

    def test_n_deaths(self):
        ages = np.array([25.0, 35.0, 45.0, 55.0])
        res = years_of_life_lost(ages)
        assert res["n_deaths"] == 4
