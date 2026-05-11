"""Tests for morie.fn.itval — item validity index."""

import numpy as np
import pandas as pd
from morie.fn.itval import item_validity_index


class TestItemValidityIndex:
    def test_returns_dataframe(self, mapq_df, rng):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        criterion = rng.standard_normal(len(mapq_df))
        result = item_validity_index(mapq_df[items], criterion)
        assert isinstance(result, pd.DataFrame)
        assert "validity_index" in result.columns
        assert len(result) == len(items)

    def test_vi_equals_r_times_sd(self, mapq_df, rng):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        criterion = rng.standard_normal(len(mapq_df))
        result = item_validity_index(mapq_df[items], criterion)
        for _, row in result.iterrows():
            expected = row["r_criterion"] * row["sd"]
            assert abs(row["validity_index"] - expected) < 1e-10

    def test_length_mismatch_raises(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        with __import__("pytest").raises(ValueError):
            item_validity_index(mapq_df[items], np.array([1, 2, 3]))

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 3))
        crit = rng.standard_normal(50)
        result = item_validity_index(data, crit)
        assert len(result) == 3
