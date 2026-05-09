"""Tests for moirais.fn.swchr -- Star Wars character summary."""

import pandas as pd
from moirais.fn.swchr import sw_character_summary, swchr
from moirais.fn._containers import DescriptiveResult


class TestSwchr:
    def test_alias(self):
        assert swchr is sw_character_summary

    def test_basic(self):
        df = pd.DataFrame({
            "name": ["Luke", "Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius", "Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius", "Leia", "Han"],
            "height": [172, 202, 66, 150, 180],
            "mass": [77, 136, 17, 49, 80],
        })
        result = sw_character_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 5
        assert "mean_height" in result.value

    def test_custom_cols(self):
        df = pd.DataFrame({"n": ["A", "B"], "h": [100, 200], "m": [50, 80]})
        result = sw_character_summary(df, name_col="n", height_col="h", mass_col="m")
        assert result.value["count"] == 2
