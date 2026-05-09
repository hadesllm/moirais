"""Tests for moirais.fn.otabl — Table 1 (baseline characteristics)."""

import pandas as pd
from moirais.fn.otabl import otis_table1


class TestOtisTable1:
    def test_returns_dataframe(self, otis_df):
        result = otis_table1(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = otis_table1(otis_df)
        for col in ("variable", "group_0", "group_1", "pval", "test"):
            assert col in result.columns

    def test_has_rows(self, otis_df):
        result = otis_table1(otis_df)
        assert len(result) > 0

    def test_pval_range(self, otis_df):
        result = otis_table1(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()
