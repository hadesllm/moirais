"""Tests for morie.fn.osumm — Full summary table."""

import pandas as pd
from morie.fn.osumm import otis_summary_table


class TestOtisSummaryTable:
    def test_returns_dataframe(self, otis_df):
        result = otis_summary_table(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_column(self, otis_df):
        result = otis_summary_table(otis_df)
        assert len(result) == len(otis_df.columns)

    def test_has_column_names(self, otis_df):
        result = otis_summary_table(otis_df)
        assert "column" in result.columns
        assert "dtype" in result.columns

    def test_numeric_has_mean(self, otis_df):
        result = otis_summary_table(otis_df)
        num_rows = result[result["dtype"] == "numeric"]
        assert "mean" in result.columns
        assert num_rows["mean"].notna().any()
