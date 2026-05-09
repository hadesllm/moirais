"""Tests for moirais.fn.ohet1 — Heterogeneous effects test by region."""

import numpy as np
from moirais.fn.ohet1 import otis_het_region


class TestOtisHetRegion:
    def test_returns_dict(self, otis_df):
        result = otis_het_region(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_het_region(otis_df)
        for k in ("f_stat", "pval", "df1", "df2", "n", "regions", "interaction_coefs"):
            assert k in result

    def test_f_stat_positive(self, otis_df):
        result = otis_het_region(otis_df)
        assert result["f_stat"] >= 0

    def test_pval_range(self, otis_df):
        result = otis_het_region(otis_df)
        assert 0 <= result["pval"] <= 1

    def test_interaction_coefs_match_regions(self, otis_df):
        result = otis_het_region(otis_df)
        assert len(result["interaction_coefs"]) == len(result["regions"])
