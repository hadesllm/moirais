"""Tests for morie.fn.cmptr — compliance trend."""

import pandas as pd
from morie.fn.cmptr import compliance_trend


class TestComplianceTrend:
    def test_returns_dataframe(self, otis_df):
        result = compliance_trend(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted(self, otis_df):
        result = compliance_trend(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_rate_bounded(self, otis_df):
        result = compliance_trend(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "flag", "end_fiscal_year": "yr"})
        result = compliance_trend(df, flag_col="flag", year_col="yr")
        assert len(result) > 0
