"""Tests for moirais.fn.tpsdiv — division compare."""

import pytest
import pandas as pd
from moirais.fn.tpsdiv import tps_division_compare
from moirais.fn._containers import DescriptiveResult


class TestDivisionCompare:
    def test_basic(self):
        df = pd.DataFrame({"division": ["D1"] * 5 + ["D2"] * 5, "crime_rate": [10, 12, 11, 13, 14, 5, 6, 4, 7, 5]})
        r = tps_division_compare(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["highest"] == "D1"

    def test_single_division(self):
        df = pd.DataFrame({"division": ["D1"] * 3, "crime_rate": [10, 10, 10]})
        r = tps_division_compare(df)
        assert r.extra["n_divisions"] == 1
