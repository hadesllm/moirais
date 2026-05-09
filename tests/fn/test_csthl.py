"""Tests for moirais.fn.csthl — custody health access."""

import pandas as pd
from moirais.fn.csthl import custody_health_access


class TestCustodyHealthAccess:
    def test_returns_dataframe(self, otis_df):
        result = custody_health_access(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_health_access(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_groups_match(self, otis_df):
        result = custody_health_access(otis_df)
        assert set(result["region"]) == set(otis_df["region"].unique())

    def test_custom_cols(self, otis_df):
        result = custody_health_access(otis_df, group_col="facility_type")
        assert len(result) > 0
