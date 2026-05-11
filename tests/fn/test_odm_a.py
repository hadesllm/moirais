"""Tests for morie.fn.odm_a — Demographic profile per age group."""

import pandas as pd
from morie.fn.odm_a import otis_demo_age


class TestOtisDemoAge:
    def test_returns_dataframe(self, otis_df):
        result = otis_demo_age(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_age(self, otis_df):
        result = otis_demo_age(otis_df)
        assert len(result) == otis_df["age_group"].nunique()

    def test_has_outcome_stats(self, otis_df):
        result = otis_demo_age(otis_df)
        assert "outcome_mean" in result.columns
        assert "outcome_sd" in result.columns
