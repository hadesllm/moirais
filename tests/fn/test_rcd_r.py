"""Tests for morie.fn.rcd_r — recidivism by region."""

import pandas as pd
from morie.fn.rcd_r import recidivism_by_region, rcd_r


class TestRecidivismByRegion:
    def test_returns_dataframe(self, otis_df):
        result = recidivism_by_region(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "region" in result.columns
        assert "rate" in result.columns

    def test_all_regions_present(self, otis_df):
        result = recidivism_by_region(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_rates_between_0_and_1(self, otis_df):
        result = recidivism_by_region(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()
