"""Tests for morie.fn.swflm -- Solar System mission summary."""

import pandas as pd
from morie.fn.swflm import solar_mission_summary, swflm
from morie.fn._containers import DescriptiveResult


class TestSwflm:
    def test_alias(self):
        assert swflm is solar_mission_summary

    def test_basic(self):
        df = pd.DataFrame({
            "name": ["Sputnik 1", "Apollo 11", "Voyager 1"],
            "launch_year": [1957, 1969, 1977],
        })
        result = solar_mission_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 3
        assert result.value["earliest"] == 1957
        assert result.value["latest"] == 1977

    def test_span(self):
        df = pd.DataFrame({"name": ["A", "B"], "launch_year": [2000, 2010]})
        result = solar_mission_summary(df)
        assert result.value["span_years"] == 10
