"""Tests for moirais.fn.odesc — OTIS demographic summary."""

import pandas as pd
from moirais.fn.odesc import otis_demographic_summary


class TestOtisDemographicSummary:
    def test_returns_dataframe(self, otis_df):
        result = otis_demographic_summary(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = otis_demographic_summary(otis_df)
        assert "count" in result.columns

    def test_total_matches(self, otis_df):
        result = otis_demographic_summary(otis_df)
        assert result["count"].sum() == len(otis_df)

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"region": "r", "age_group": "a", "gender": "g"})
        result = otis_demographic_summary(df, region_col="r", age_col="a", gender_col="g")
        assert len(result) > 0
