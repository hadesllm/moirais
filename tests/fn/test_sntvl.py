"""Tests for morie.fn.sntvl — sentence volatility."""

import pandas as pd
from morie.fn.sntvl import sentence_volatility, sntvl


class TestSentenceVolatility:
    def test_returns_dataframe(self, otis_df):
        result = sentence_volatility(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "cv" in result.columns

    def test_only_multi_placement(self, otis_df):
        result = sentence_volatility(otis_df)
        assert (result["n_placements"] >= 2).all()

    def test_range_nonneg(self, otis_df):
        result = sentence_volatility(otis_df)
        assert (result["range_sentence"] >= 0).all()
