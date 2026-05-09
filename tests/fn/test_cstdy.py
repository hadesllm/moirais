"""Tests for moirais.fn.cstdy — custody days per individual."""

import pandas as pd
from moirais.fn.cstdy import custody_days


class TestCustodyDays:
    def test_returns_dataframe(self, otis_df):
        result = custody_days(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = custody_days(otis_df)
        assert "total_days" in result.columns
        assert "n_records" in result.columns

    def test_unique_ids(self, otis_df):
        result = custody_days(otis_df)
        assert result["unique_individual_id"].nunique() == otis_df["unique_individual_id"].nunique()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"sentence_days": "days", "unique_individual_id": "pid"})
        result = custody_days(df, sent_col="days", id_col="pid")
        assert len(result) > 0
