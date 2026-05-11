"""Tests for morie.fn.ohist — Histogram bin counts."""

import pandas as pd
from morie.fn.ohist import otis_histogram_data


class TestOtisHistogramData:
    def test_returns_dataframe(self, otis_df):
        result = otis_histogram_data(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_bins(self, otis_df):
        result = otis_histogram_data(otis_df, bins=10)
        assert len(result) == 10

    def test_columns(self, otis_df):
        result = otis_histogram_data(otis_df)
        for col in ("bin_lo", "bin_hi", "count", "density"):
            assert col in result.columns

    def test_counts_sum(self, otis_df):
        result = otis_histogram_data(otis_df)
        assert result["count"].sum() == otis_df["sentence_days"].notna().sum()

    def test_bins_ordered(self, otis_df):
        result = otis_histogram_data(otis_df)
        assert (result["bin_lo"] < result["bin_hi"]).all()
