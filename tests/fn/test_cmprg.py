"""Tests for morie.fn.cmprg — compliance by region."""

import pandas as pd
from morie.fn.cmprg import compliance_by_region


class TestComplianceByRegion:
    def test_returns_dataframe(self, otis_df):
        result = compliance_by_region(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_regions_match(self, otis_df):
        result = compliance_by_region(otis_df)
        assert set(result["region"]) == set(otis_df["region"].unique())

    def test_rate_bounded(self, otis_df):
        result = compliance_by_region(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "f", "region": "r"})
        result = compliance_by_region(df, flag_col="f", region_col="r")
        assert len(result) > 0
