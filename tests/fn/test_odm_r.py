"""Tests for moirais.fn.odm_r — Demographic profile per region."""

import pandas as pd
from moirais.fn.odm_r import otis_demo_region


class TestOtisDemoRegion:
    def test_returns_dataframe(self, otis_df):
        result = otis_demo_region(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_region(self, otis_df):
        result = otis_demo_region(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_has_n_column(self, otis_df):
        result = otis_demo_region(otis_df)
        assert "n" in result.columns
        assert (result["n"] > 0).all()

    def test_custom_col(self, otis_df):
        df = otis_df.rename(columns={"region": "area"})
        result = otis_demo_region(df, region_col="area")
        assert len(result) > 0
