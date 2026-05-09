"""Tests for moirais.fn.ittab — full item analysis table."""

import numpy as np
import pandas as pd
from moirais.fn.ittab import item_table


class TestItemTable:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_table(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(items)

    def test_expected_columns(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_table(mapq_df[items])
        expected_cols = {"item", "mean", "sd", "difficulty", "r_corrected",
                         "alpha_if_deleted", "floor_pct", "ceiling_pct", "skewness"}
        assert expected_cols.issubset(set(result.columns))

    def test_difficulty_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_table(mapq_df[items])
        assert (result["difficulty"] >= 0).all()
        assert (result["difficulty"] <= 1).all()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 5))
        result = item_table(data)
        assert len(result) == 5
        assert result["item"].tolist() == [f"i{i}" for i in range(5)]

    def test_binary_items(self, mapq_binary_df):
        result = item_table(mapq_binary_df)
        assert len(result) == 10
