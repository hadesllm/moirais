"""Tests for morie.fn.sntgn — sentence by gender."""

import pandas as pd
from morie.fn.sntgn import sentence_by_gender, sntgn


class TestSentenceByGender:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_gender(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "gender" in result.columns

    def test_two_groups(self, otis_df):
        result = sentence_by_gender(otis_df)
        assert len(result) == 2

    def test_n_positive(self, otis_df):
        result = sentence_by_gender(otis_df)
        assert (result["n"] > 0).all()
