"""Tests for moirais.fn.sntrl — sentence trends over years."""

import pandas as pd
from moirais.fn.sntrl import sentence_by_year, sntrl


class TestSentenceByYear:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_year(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "mean_sentence" in result.columns

    def test_years_sorted(self, otis_df):
        result = sentence_by_year(otis_df)
        years = result["end_fiscal_year"].values
        assert (years[1:] >= years[:-1]).all()

    def test_n_sums(self, otis_df):
        result = sentence_by_year(otis_df)
        assert result["n"].sum() == len(otis_df)
