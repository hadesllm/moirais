"""Tests for morie.fn.sntsr — proportion of sentence served."""

from morie.fn.sntsr import sentence_served, sntsr
import numpy as np


class TestSentenceServed:
    def test_returns_dict_no_time_col(self, otis_df):
        result = sentence_served(otis_df)
        assert isinstance(result, dict)
        assert "n" in result
        assert np.isnan(result["mean_proportion"])

    def test_returns_dict_with_time_col(self, otis_df):
        otis_df = otis_df.copy()
        otis_df["time_served"] = otis_df["sentence_days"] * 0.7
        result = sentence_served(otis_df, time_col="time_served")
        assert 0.0 <= result["mean_proportion"] <= 1.0

    def test_n_positive(self, otis_df):
        result = sentence_served(otis_df)
        assert result["n"] > 0
