"""Tests for morie.fn.cfami -- CFA modification indices."""

import pandas as pd
from morie.fn.cfami import cfa_modindex
from morie.fn._mapq_const import SUBSCALES


class TestCfaModIndex:

    def test_returns_dataframe(self, mapq_df):
        result = cfa_modindex(mapq_df, SUBSCALES)
        assert isinstance(result, pd.DataFrame)
        assert "param" in result.columns
        assert "mi" in result.columns

    def test_top_n_rows(self, mapq_df):
        result = cfa_modindex(mapq_df, SUBSCALES, top_n=5)
        assert len(result) <= 5

    def test_mi_nonnegative(self, mapq_df):
        result = cfa_modindex(mapq_df, SUBSCALES)
        assert (result["mi"] >= 0).all()
