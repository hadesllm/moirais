"""Tests for morie.fn.rcdrt — recidivism trend over years."""

import pandas as pd
from morie.fn.rcdrt import recidivism_trend, rcdrt


class TestRecidivismTrend:
    def test_returns_dataframe(self, otis_df):
        result = recidivism_trend(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "year" in result.columns
        assert "rate" in result.columns

    def test_years_sorted(self, otis_df):
        result = recidivism_trend(otis_df)
        years = result["year"].values
        assert (years[1:] >= years[:-1]).all()

    def test_covers_all_years(self, otis_df):
        result = recidivism_trend(otis_df)
        assert len(result) == otis_df["end_fiscal_year"].nunique()
