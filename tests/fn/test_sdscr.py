"""Tests for morie.fn.sdscr -- discriminant validity."""

import pandas as pd
from morie.fn.sdscr import subscale_discriminant


class TestSubscaleDiscriminant:

    def test_returns_dataframe(self, mapq_df):
        result = subscale_discriminant(mapq_df)
        assert isinstance(result, pd.DataFrame)
        assert "pass" in result.columns

    def test_six_pairs(self, mapq_df):
        result = subscale_discriminant(mapq_df)
        assert len(result) == 6  # C(4,2) = 6

    def test_sqrt_ave_positive(self, mapq_df):
        result = subscale_discriminant(mapq_df)
        assert (result["sqrt_ave_1"] >= 0).all()
        assert (result["sqrt_ave_2"] >= 0).all()
