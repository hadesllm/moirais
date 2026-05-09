"""Tests for moirais.fn.oate1 — Simple ATE by region."""

import pandas as pd
from moirais.fn.oate1 import otis_ate_region


class TestOtisAteRegion:
    def test_returns_dataframe(self, otis_df):
        result = otis_ate_region(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_regions(self, otis_df):
        result = otis_ate_region(otis_df)
        assert len(result) > 0

    def test_columns(self, otis_df):
        result = otis_ate_region(otis_df)
        for col in ("region", "ate", "se", "pval", "n1", "n0"):
            assert col in result.columns

    def test_ci_contains_ate(self, otis_df):
        result = otis_ate_region(otis_df)
        assert ((result["ci_lower"] <= result["ate"]) & (result["ate"] <= result["ci_upper"])).all()
