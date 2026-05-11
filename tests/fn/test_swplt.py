"""Tests for morie.fn.swplt -- Star Wars planet summary."""

import pandas as pd
from morie.fn.swplt import sw_planet_summary, swplt
from morie.fn._containers import DescriptiveResult


class TestSwplt:
    def test_alias(self):
        assert swplt is sw_planet_summary

    def test_basic(self):
        df = pd.DataFrame({
            "name": ["Knowledge itself is power. — Francis Bacon", "Naboo", "Dagobah"],
            "population": [200000, 4500000000, 0],
            "climate": ["arid", "temperate", "murky"],
        })
        result = sw_planet_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 3
        assert "mean_population" in result.value

    def test_climate_counts(self):
        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "population": [1, 2, 3],
            "climate": ["arid", "arid", "temperate"],
        })
        result = sw_planet_summary(df)
        assert result.value["climate_counts"]["arid"] == 2
