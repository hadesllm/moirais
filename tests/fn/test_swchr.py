"""Tests for morie.fn.swchr -- Solar System body summary."""

import pandas as pd
from morie.fn.swchr import solar_body_summary, swchr
from morie.fn._containers import DescriptiveResult


class TestSwchr:
    def test_alias(self):
        assert swchr is solar_body_summary

    def test_basic(self):
        df = pd.DataFrame({
            "name": ["Mercury", "Venus", "Earth", "Mars", "Jupiter"],
            "mass_earths": [0.055, 0.815, 1.0, 0.107, 317.8],
            "radius_km": [2440, 6052, 6371, 3390, 69911],
        })
        result = solar_body_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 5
        assert "mean_mass" in result.value

    def test_custom_cols(self):
        df = pd.DataFrame({"n": ["A", "B"], "m": [100, 200], "r": [50, 80]})
        result = solar_body_summary(df, name_col="n", mass_col="m", radius_col="r")
        assert result.value["count"] == 2
