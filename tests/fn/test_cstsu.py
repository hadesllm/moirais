"""Tests for moirais.fn.cstsu — custody substance flag by age."""

import pandas as pd
from moirais.fn.cstsu import custody_substance


class TestCustodySubstance:
    def test_returns_dataframe(self, otis_df):
        result = custody_substance(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_substance(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_groups_match(self, otis_df):
        result = custody_substance(otis_df)
        assert set(result["age_group"]) == set(otis_df["age_group"].unique())

    def test_custom_cols(self, otis_df):
        result = custody_substance(otis_df, flag_col="D", group_col="gender")
        assert len(result) > 0
