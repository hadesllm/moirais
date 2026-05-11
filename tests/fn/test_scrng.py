"""Tests for morie.fn.scrng — score range check."""

import numpy as np
import pandas as pd
from morie.fn.scrng import score_range_check


class TestScoreRangeCheck:
    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_range_check(mapq_df, items=items, min_val=1, max_val=5)
        assert isinstance(result, dict)
        assert "n_invalid" in result
        assert "pct_invalid" in result

    def test_valid_data_no_flags(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_range_check(mapq_df, items=items, min_val=1, max_val=5)
        assert result["n_invalid"] == 0
        assert len(result["flagged_rows"]) == 0

    def test_detects_out_of_range(self):
        df = pd.DataFrame({"a": [1, 2, 6, 4], "b": [1, 0, 3, 5]})
        result = score_range_check(df, min_val=1, max_val=5)
        assert result["n_invalid"] == 2  # 6 and 0
        assert 1 in result["flagged_rows"]  # row with 0
        assert 2 in result["flagged_rows"]  # row with 6

    def test_flagged_items(self):
        df = pd.DataFrame({"ok": [1, 2, 3], "bad": [1, 2, 99]})
        result = score_range_check(df, min_val=1, max_val=5)
        assert "bad" in result["flagged_items"]
        assert "ok" not in result["flagged_items"]

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 5))
        result = score_range_check(data, min_val=1, max_val=5)
        assert result["n_invalid"] == 0
