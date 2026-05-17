"""Tests for morie.fn.swplt -- Solar System orbit summary."""

import pandas as pd
from morie.fn.swplt import solar_orbit_summary, swplt
from morie.fn._containers import DescriptiveResult


class TestSwplt:
    def test_alias(self):
        assert swplt is solar_orbit_summary

    def test_basic(self):
        df = pd.DataFrame({
            "name": ["Mercury", "Earth", "Jupiter"],
            "orbital_period_days": [88.0, 365.25, 4332.6],
            "moons": [0, 1, 95],
        })
        result = solar_orbit_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 3
        assert "mean_period_days" in result.value

    def test_total_moons(self):
        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "orbital_period_days": [1.0, 2.0, 3.0],
            "moons": [0, 2, 4],
        })
        result = solar_orbit_summary(df)
        assert result.value["total_moons"] == 6
