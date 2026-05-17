"""Tests for morie.fn.swdb -- Solar System demo dataset loader."""

import pytest
from morie.fn.swdb import load_solar_system, load_solar_system_result, swdb
from morie.fn._containers import DescriptiveResult


class TestSwdb:
    def test_planets_columns(self):
        df = load_solar_system("planets")
        assert "name" in df.columns
        assert "mass_earths" in df.columns
        assert "radius_km" in df.columns
        assert "orbital_period_days" in df.columns

    def test_all_tables(self):
        for table in ("planets", "moons", "missions"):
            df = load_solar_system(table)
            assert len(df) > 0

    def test_unknown_table(self):
        with pytest.raises(ValueError, match="Unknown table"):
            load_solar_system("asteroids")

    def test_result_alias(self):
        assert swdb is load_solar_system_result
        result = swdb("planets")
        assert isinstance(result, DescriptiveResult)
