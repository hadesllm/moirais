"""Tests for moirais.fn.sitdl -- per-item detail within subscale."""

import pandas as pd
from moirais.fn.sitdl import subscale_item_detail


class TestSubscaleItemDetail:

    def test_returns_dataframe(self, mapq_df):
        result = subscale_item_detail(mapq_df, "EE")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_has_expected_columns(self, mapq_df):
        result = subscale_item_detail(mapq_df, "EE")
        for col in ("item", "mean", "sd", "skew", "corrected_item_total_r", "alpha_if_deleted"):
            assert col in result.columns

    def test_custom_items(self, mapq_df):
        result = subscale_item_detail(mapq_df, "custom", items=["EA1", "EA2", "EA3"])
        assert len(result) == 3
