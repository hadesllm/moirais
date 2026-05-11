"""Tests for morie.fn.itsel — item selection / flagging."""

import numpy as np
import pandas as pd
from morie.fn.itsel import item_select


class TestItemSelect:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_select(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "flagged" in result.columns
        assert len(result) == len(items)

    def test_flag_columns_boolean(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_select(mapq_df[items])
        assert result["flag_r"].dtype == bool
        assert result["flag_floor"].dtype == bool
        assert result["flag_ceiling"].dtype == bool

    def test_constant_item_flagged(self):
        rng = np.random.default_rng(42)
        data = np.column_stack([np.ones(100), rng.integers(1, 6, 100)])
        result = item_select(data, min_r=0.3)
        # Constant item should be flagged for low discrimination
        assert result.loc[0, "flagged"]

    def test_strict_threshold(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        strict = item_select(mapq_df[items], min_r=0.9)
        lenient = item_select(mapq_df[items], min_r=0.1)
        # Strict should flag at least as many as lenient
        assert strict["flagged"].sum() >= lenient["flagged"].sum()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 5))
        result = item_select(data)
        assert len(result) == 5
