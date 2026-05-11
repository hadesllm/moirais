"""Tests for morie.fn.cstlk — custody lockdown frequency."""

import pandas as pd
from morie.fn.cstlk import custody_lockdown_freq


class TestCustodyLockdownFreq:
    def test_returns_dataframe(self, otis_df):
        result = custody_lockdown_freq(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted(self, otis_df):
        result = custody_lockdown_freq(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_rate_bounded(self, otis_df):
        result = custody_lockdown_freq(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "ev", "end_fiscal_year": "yr"})
        result = custody_lockdown_freq(df, event_col="ev", year_col="yr")
        assert len(result) > 0
