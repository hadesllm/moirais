"""Tests for moirais.fn.cstmh — custody mental health trend."""

import pandas as pd
from moirais.fn.cstmh import custody_mental_health


class TestCustodyMentalHealth:
    def test_returns_dataframe(self, otis_df):
        result = custody_mental_health(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_year(self, otis_df):
        result = custody_mental_health(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_rate_bounded(self, otis_df):
        result = custody_mental_health(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"alert_mental_health": "mh", "end_fiscal_year": "yr"})
        result = custody_mental_health(df, mh_col="mh", year_col="yr")
        assert len(result) > 0
