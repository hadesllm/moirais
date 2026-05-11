"""Tests for morie.fn.ofreq — Frequency table."""

import pandas as pd
from morie.fn.ofreq import otis_freq_table


class TestOtisFreqTable:
    def test_returns_dataframe(self, otis_df):
        result = otis_freq_table(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = otis_freq_table(otis_df)
        for col in ("value", "count", "proportion", "cumulative_proportion"):
            assert col in result.columns

    def test_proportions_sum_to_one(self, otis_df):
        result = otis_freq_table(otis_df)
        assert abs(result["proportion"].sum() - 1.0) < 0.01

    def test_cumulative_ends_at_one(self, otis_df):
        result = otis_freq_table(otis_df)
        assert abs(result["cumulative_proportion"].iloc[-1] - 1.0) < 0.01

    def test_custom_col(self, otis_df):
        result = otis_freq_table(otis_df, col="gender")
        assert len(result) == otis_df["gender"].nunique()
