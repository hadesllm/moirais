"""Tests for morie.fn.supol -- polysubstance use."""

import pandas as pd
from morie.fn.supol import polysubstance


class TestPolysubstance:
    def test_basic(self):
        df = pd.DataFrame({
            "alcohol": [1, 1, 0, 1],
            "cannabis": [1, 0, 1, 1],
            "opioid": [0, 0, 1, 1],
        })
        res = polysubstance(df)
        assert res.name == "polysubstance"
        assert res.extra["poly_counts"]["poly_3plus"] == 1

    def test_no_use(self):
        df = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
        res = polysubstance(df)
        assert res.extra["poly_counts"]["none"] == 2
