"""Tests for moirais.fn.itrel — item reliability index."""

import numpy as np
import pandas as pd
from moirais.fn.itrel import item_reliability_index


class TestItemReliabilityIndex:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_reliability_index(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "reliability_index" in result.columns
        assert len(result) == len(items)

    def test_ri_equals_r_times_sd(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_reliability_index(mapq_df[items])
        for _, row in result.iterrows():
            expected = row["r_corrected"] * row["sd"]
            assert abs(row["reliability_index"] - expected) < 1e-10

    def test_positive_for_correlated(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_reliability_index(mapq_df[items])
        assert (result["reliability_index"] > 0).all()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 4))
        result = item_reliability_index(data)
        assert len(result) == 4
