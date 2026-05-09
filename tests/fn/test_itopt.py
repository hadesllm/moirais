"""Tests for moirais.fn.itopt — item option frequencies."""

import numpy as np
import pandas as pd
from moirais.fn.itopt import item_option_freq


class TestItemOptionFreq:
    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_option_freq(mapq_df[items])
        assert isinstance(result, dict)
        assert len(result) == len(items)

    def test_each_value_is_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_option_freq(mapq_df[items])
        for key, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert "count" in df.columns
            assert "pct" in df.columns

    def test_counts_sum_to_n(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_option_freq(mapq_df[items])
        for key, df in result.items():
            assert df["count"].sum() == len(mapq_df)

    def test_pct_sums_to_100(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_option_freq(mapq_df[items])
        for key, df in result.items():
            assert abs(df["pct"].sum() - 100.0) < 0.01

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 3))
        result = item_option_freq(data)
        assert len(result) == 3
