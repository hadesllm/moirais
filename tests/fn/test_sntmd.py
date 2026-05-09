"""Tests for moirais.fn.sntmd — sentence by group."""

import pandas as pd
from moirais.fn.sntmd import sentence_by_group, sntmd


class TestSentenceByGroup:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_group(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "median_sentence" in result.columns

    def test_groups_match(self, otis_df):
        result = sentence_by_group(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_n_sums(self, otis_df):
        result = sentence_by_group(otis_df)
        assert result["n"].sum() == len(otis_df)
