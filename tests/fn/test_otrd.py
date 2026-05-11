"""Tests for morie.fn.otrd — OTIS trend summary."""

import pandas as pd
from morie.fn.otrd import otis_trend_summary


class TestOtisTrendSummary:
    def test_returns_dataframe(self, otis_df):
        result = otis_trend_summary(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = otis_trend_summary(otis_df)
        for col in ("mean", "std", "n", "change", "pct_change"):
            assert col in result.columns

    def test_sorted(self, otis_df):
        result = otis_trend_summary(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "metric", "end_fiscal_year": "yr"})
        result = otis_trend_summary(df, metric_col="metric", year_col="yr")
        assert len(result) > 0
