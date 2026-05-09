"""Tests for moirais.fn.odml3 — DML ATE by age group."""

import pandas as pd
from moirais.fn.odml3 import otis_dml_age


class TestOtisDmlAge:
    def test_returns_dataframe(self, otis_df):
        result = otis_dml_age(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_age_groups(self, otis_df):
        result = otis_dml_age(otis_df)
        assert len(result) == otis_df["age_group"].nunique()

    def test_columns(self, otis_df):
        result = otis_dml_age(otis_df)
        for col in ("age_group", "ate", "se", "pval", "n"):
            assert col in result.columns

    def test_pval_range(self, otis_df):
        result = otis_dml_age(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()
