"""Tests for morie.fn.cstpg — custody program participation rate."""

import pandas as pd
from morie.fn.cstpg import custody_program_rate


class TestCustodyProgramRate:
    def test_returns_dataframe(self, otis_df):
        result = custody_program_rate(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_program_rate(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_groups_match(self, otis_df):
        result = custody_program_rate(otis_df)
        assert set(result["region"]) == set(otis_df["region"].unique())

    def test_custom_cols(self, otis_df):
        result = custody_program_rate(otis_df, flag_col="D", group_col="facility_type")
        assert len(result) > 0
