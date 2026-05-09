"""Tests for moirais.fn.itdsc — item discrimination (corrected item-total r)."""

import numpy as np
import pandas as pd
from moirais.fn.itdsc import item_discrimination_all


class TestItemDiscriminationAll:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_discrimination_all(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "r_corrected" in result.columns
        assert len(result) == len(items)

    def test_correlated_items_positive(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_discrimination_all(mapq_df[items])
        assert (result["r_corrected"] > 0).all()

    def test_constant_item_zero(self):
        data = np.column_stack([np.ones(50), np.random.default_rng(42).integers(1, 5, 50)])
        result = item_discrimination_all(data)
        assert result.loc[0, "r_corrected"] == 0.0

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 5))
        result = item_discrimination_all(data)
        assert len(result) == 5

    def test_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_discrimination_all(mapq_df[items])
        assert (result["r_corrected"] >= -1).all()
        assert (result["r_corrected"] <= 1).all()
