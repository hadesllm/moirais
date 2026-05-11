"""Tests for morie.fn.csttr — custody transfer rate."""

import pandas as pd
from morie.fn.csttr import custody_transfer_rate


class TestCustodyTransferRate:
    def test_returns_dataframe(self, otis_df):
        result = custody_transfer_rate(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_transfer_rate(otis_df)
        assert (result["transfer_rate"] >= 0).all()
        assert (result["transfer_rate"] <= 1).all()

    def test_sorted(self, otis_df):
        result = custody_transfer_rate(otis_df)
        years = result["end_fiscal_year"].tolist()
        assert years == sorted(years)

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"region": "reg", "unique_individual_id": "pid", "end_fiscal_year": "yr"})
        result = custody_transfer_rate(df, region_col="reg", id_col="pid", year_col="yr")
        assert len(result) > 0
