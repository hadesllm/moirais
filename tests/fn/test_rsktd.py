"""Tests for morie.fn.rsktd — risk score trend over years."""

import pandas as pd
from morie.fn.rsktd import risk_trend, rsktd


class TestRiskTrend:
    def test_returns_dataframe(self, otis_df):
        result = risk_trend(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "mean_score" in result.columns

    def test_years_sorted(self, otis_df):
        result = risk_trend(otis_df)
        years = result["end_fiscal_year"].values
        assert (years[1:] >= years[:-1]).all()

    def test_n_per_year(self, otis_df):
        result = risk_trend(otis_df)
        assert result["n"].sum() == len(otis_df)
