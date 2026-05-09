"""Tests for moirais.fn.oatt1 — ATT by region via IPW."""

import pandas as pd
from moirais.fn.oatt1 import otis_att_region


class TestOtisAttRegion:
    def test_returns_dataframe(self, otis_df):
        result = otis_att_region(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_regions(self, otis_df):
        result = otis_att_region(otis_df)
        assert len(result) > 0

    def test_columns(self, otis_df):
        result = otis_att_region(otis_df)
        for col in ("region", "att", "se", "pval", "n"):
            assert col in result.columns

    def test_pval_range(self, otis_df):
        result = otis_att_region(otis_df)
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()
