"""Tests for morie.fn.sntrg — sentence by region."""

import pandas as pd
from morie.fn.sntrg import sentence_by_region, sntrg


class TestSentenceByRegion:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_region(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "region" in result.columns

    def test_n_regions(self, otis_df):
        result = sentence_by_region(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_has_median(self, otis_df):
        result = sentence_by_region(otis_df)
        assert "median_sentence" in result.columns
