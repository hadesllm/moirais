"""Tests for morie.fn.cstoc — custody occupancy."""

import pandas as pd
from morie.fn.cstoc import custody_occupancy


class TestCustodyOccupancy:
    def test_returns_dataframe(self, otis_df):
        result = custody_occupancy(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = custody_occupancy(otis_df)
        assert "count" in result.columns

    def test_total_matches(self, otis_df):
        result = custody_occupancy(otis_df)
        assert result["count"].sum() == len(otis_df)

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"end_fiscal_year": "yr", "facility_type": "fac"})
        result = custody_occupancy(df, year_col="yr", facility_col="fac")
        assert len(result) > 0
