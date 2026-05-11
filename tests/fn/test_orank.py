"""Tests for morie.fn.orank — OTIS rank regions."""

import pandas as pd
from morie.fn.orank import otis_rank_regions


class TestOtisRankRegions:
    def test_returns_dataframe(self, otis_df):
        result = otis_rank_regions(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rank_column(self, otis_df):
        result = otis_rank_regions(otis_df)
        assert "rank" in result.columns
        assert list(result["rank"]) == list(range(1, len(result) + 1))

    def test_sorted_descending(self, otis_df):
        result = otis_rank_regions(otis_df)
        means = result["mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_regions_match(self, otis_df):
        result = otis_rank_regions(otis_df)
        assert set(result["region"]) == set(otis_df["region"].unique())

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "m", "region": "r"})
        result = otis_rank_regions(df, metric_col="m", region_col="r")
        assert len(result) > 0
