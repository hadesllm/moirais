"""Tests for morie.fn.irtdp — distractor analysis."""

import numpy as np
import pandas as pd
from morie.fn.irtdp import irt_distractor


class TestIrtDistractor:
    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = irt_distractor(mapq_df[items])
        assert isinstance(result, dict)
        assert len(result) == len(items)

    def test_each_item_has_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = irt_distractor(mapq_df[items])
        for key, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert "option" in df.columns

    def test_with_answer_key(self, mapq_binary_df):
        key = {f"item_{i+1}": 1 for i in range(10)}
        result = irt_distractor(mapq_binary_df, key=key)
        for item, df in result.items():
            if item in key:
                assert "correct" in df.columns

    def test_proportions_sum_to_one(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = irt_distractor(mapq_df[items], n_groups=3)
        for item, df in result.items():
            for g_col in [c for c in df.columns if c.startswith("group_")]:
                total = df[g_col].sum()
                assert abs(total - 1.0) < 0.01

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 5))
        result = irt_distractor(data)
        assert len(result) == 5
