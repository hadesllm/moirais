"""Tests for moirais.fn.opair — Pairwise group comparisons."""

import pandas as pd
from moirais.fn.opair import otis_pairwise_compare


class TestOtisPairwiseCompare:
    def test_returns_dataframe(self, otis_df):
        result = otis_pairwise_compare(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_pairs(self, otis_df):
        result = otis_pairwise_compare(otis_df)
        n_groups = otis_df["region"].nunique()
        expected = n_groups * (n_groups - 1) // 2
        assert len(result) == expected

    def test_columns(self, otis_df):
        result = otis_pairwise_compare(otis_df)
        for col in ("group_1", "group_2", "diff", "statistic", "pval", "pval_adj"):
            assert col in result.columns

    def test_bonferroni_adj(self, otis_df):
        result = otis_pairwise_compare(otis_df)
        assert (result["pval_adj"] >= result["pval"]).all()
        assert (result["pval_adj"] <= 1.0).all()

    def test_mann_whitney(self, otis_df):
        result = otis_pairwise_compare(otis_df, method="mw")
        assert len(result) > 0
