"""Tests for moirais.fn.ocorr — Correlation matrix."""

import pandas as pd
from moirais.fn.ocorr import otis_correlation


class TestOtisCorrelation:
    def test_returns_dataframe(self, otis_df):
        result = otis_correlation(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_square_matrix(self, otis_df):
        result = otis_correlation(otis_df)
        assert result.shape[0] == result.shape[1]

    def test_diagonal_is_one(self, otis_df):
        result = otis_correlation(otis_df)
        for col in result.columns:
            assert abs(result.loc[col, col] - 1.0) < 1e-10

    def test_custom_cols(self, otis_df):
        result = otis_correlation(otis_df, cols=["Y", "D", "sentence_days"])
        assert result.shape == (3, 3)

    def test_spearman(self, otis_df):
        result = otis_correlation(otis_df, method="spearman")
        assert isinstance(result, pd.DataFrame)
