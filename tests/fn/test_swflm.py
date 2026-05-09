"""Tests for moirais.fn.swflm -- Star Wars film summary."""

import pandas as pd
from moirais.fn.swflm import sw_film_summary, swflm
from moirais.fn._containers import DescriptiveResult


class TestSwflm:
    def test_alias(self):
        assert swflm is sw_film_summary

    def test_basic(self):
        df = pd.DataFrame({
            "title": ["A New Hope", "Empire Strikes Back", "Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu"],
            "release_year": [1977, 1980, 1983],
        })
        result = sw_film_summary(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["count"] == 3
        assert result.value["earliest"] == 1977
        assert result.value["latest"] == 1983

    def test_span(self):
        df = pd.DataFrame({"title": ["A", "B"], "release_year": [2000, 2010]})
        result = sw_film_summary(df)
        assert result.value["span_years"] == 10
