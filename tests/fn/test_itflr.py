"""Tests for moirais.fn.itflr — item floor and ceiling effects."""

import numpy as np
import pandas as pd
from moirais.fn.itflr import item_floor_ceiling


class TestItemFloorCeiling:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_floor_ceiling(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "floor_pct" in result.columns
        assert "ceiling_pct" in result.columns

    def test_range_0_100(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_floor_ceiling(mapq_df[items])
        assert (result["floor_pct"] >= 0).all()
        assert (result["floor_pct"] <= 100).all()
        assert (result["ceiling_pct"] >= 0).all()
        assert (result["ceiling_pct"] <= 100).all()

    def test_constant_item(self):
        data = np.full((50, 2), 3.0)
        result = item_floor_ceiling(data)
        # All at same value = 100% floor AND 100% ceiling
        assert result["floor_pct"].iloc[0] == 100.0
        assert result["ceiling_pct"].iloc[0] == 100.0

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 4))
        result = item_floor_ceiling(data)
        assert len(result) == 4
