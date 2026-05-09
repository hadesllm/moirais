"""Tests for moirais.fn.scsum — sum score."""

import numpy as np
import pandas as pd
from moirais.fn.scsum import score_sum


class TestScoreSum:
    def test_returns_series(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_sum(mapq_df, items=items)
        assert isinstance(result, pd.Series)
        assert len(result) == len(mapq_df)

    def test_sum_correct(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = score_sum(df)
        assert result.tolist() == [5, 7, 9]

    def test_all_items(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = score_sum(mapq_df[items])
        assert (result >= 20).all()  # 20 items, min 1 each
        assert (result <= 100).all()  # 20 items, max 5 each

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 5))
        result = score_sum(data)
        assert len(result) == 50

    def test_subset_items(self, mapq_df):
        full = score_sum(mapq_df, items=["EE1", "EE2", "EE3", "EE4", "EE5"])
        part = score_sum(mapq_df, items=["EE1", "EE2"])
        assert (full >= part).all()
