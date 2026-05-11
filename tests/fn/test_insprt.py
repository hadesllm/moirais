"""Tests for morie.fn.insprt — inspection score by facility."""

import pandas as pd
from morie.fn.insprt import inspection_score


class TestInspectionScore:
    def test_returns_dataframe(self, otis_df):
        result = inspection_score(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = inspection_score(otis_df)
        assert "mean_score" in result.columns
        assert "std_score" in result.columns
        assert "n" in result.columns

    def test_facilities_match(self, otis_df):
        result = inspection_score(otis_df)
        assert set(result["facility_type"]) == set(otis_df["facility_type"].unique())

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "score", "facility_type": "fac"})
        result = inspection_score(df, score_col="score", facility_col="fac")
        assert len(result) > 0
