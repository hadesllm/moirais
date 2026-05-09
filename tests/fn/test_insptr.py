"""Tests for moirais.fn.insptr — inspection score trend."""

import pandas as pd
from moirais.fn.insptr import inspection_trend


class TestInspectionTrend:
    def test_returns_dataframe(self, otis_df):
        result = inspection_trend(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted(self, otis_df):
        result = inspection_trend(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_columns(self, otis_df):
        result = inspection_trend(otis_df)
        assert "mean_score" in result.columns
        assert "n" in result.columns

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "s", "end_fiscal_year": "yr"})
        result = inspection_trend(df, score_col="s", year_col="yr")
        assert len(result) > 0
