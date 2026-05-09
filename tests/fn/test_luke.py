"""Tests for moirais.fn.luke -- Data summary."""

import numpy as np
import pandas as pd
from moirais.fn.luke import summarize, luke
from moirais.fn._containers import DescriptiveResult


class TestLuke:
    def test_alias(self):
        assert luke is summarize

    def test_numeric_summary(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(10, 2, 100),
            "b": rng.normal(5, 1, 100),
        })
        result = summarize(df)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "Summary"
        tbl = result.value
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) == 2
        assert "mean" in tbl.columns
        assert "skew" in tbl.columns

    def test_mixed_types(self):
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
        })
        result = summarize(df)
        tbl = result.value
        assert len(tbl) == 2
        # Numeric row has mean, categorical row has n_unique
        num_row = tbl[tbl["column"] == "num"].iloc[0]
        cat_row = tbl[tbl["column"] == "cat"].iloc[0]
        assert "mean" in num_row.index
        assert "n_unique" in cat_row.index

    def test_missing_counted(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = summarize(df)
        row = result.value.iloc[0]
        assert row["missing"] == 2
        assert row["n"] == 3
