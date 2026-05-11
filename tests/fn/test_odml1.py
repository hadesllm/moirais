"""Tests for morie.fn.odml1 — DML ATE by region."""

import pandas as pd
from morie.fn.odml1 import otis_dml_region


class TestOtisDmlRegion:
    def test_returns_dataframe(self, otis_df):
        result = otis_dml_region(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_regions(self, otis_df):
        result = otis_dml_region(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_columns(self, otis_df):
        result = otis_dml_region(otis_df)
        for col in ("region", "ate", "se", "pval", "ci_lower", "ci_upper", "n"):
            assert col in result.columns

    def test_pval_range(self, otis_df):
        result = otis_dml_region(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()

    def test_custom_columns(self, otis_df):
        df = otis_df.rename(columns={"Y": "out", "D": "treat", "region": "area"})
        result = otis_dml_region(df, outcome="out", treatment="treat", region_col="area")
        assert len(result) > 0
