"""Tests for moirais.fn.odml4 — DML ATE by gender."""

import pandas as pd
from moirais.fn.odml4 import otis_dml_gender


class TestOtisDmlGender:
    def test_returns_dataframe(self, otis_df):
        result = otis_dml_gender(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_genders(self, otis_df):
        result = otis_dml_gender(otis_df)
        assert len(result) == otis_df["gender"].nunique()

    def test_columns(self, otis_df):
        result = otis_dml_gender(otis_df)
        for col in ("gender", "ate", "se", "pval", "n"):
            assert col in result.columns

    def test_pval_range(self, otis_df):
        result = otis_dml_gender(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()
