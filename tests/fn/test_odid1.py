"""Tests for morie.fn.odid1 — DiD for policy change."""

import numpy as np
from morie.fn.odid1 import otis_did_policy


class TestOtisDidPolicy:
    def test_returns_dict(self, otis_df):
        result = otis_did_policy(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_did_policy(otis_df)
        for k in ("did_estimate", "se", "pval", "n", "pre_treat", "post_treat"):
            assert k in result

    def test_did_finite(self, otis_df):
        result = otis_did_policy(otis_df)
        assert np.isfinite(result["did_estimate"])

    def test_pval_range(self, otis_df):
        result = otis_did_policy(otis_df)
        assert 0 <= result["pval"] <= 1

    def test_custom_post_year(self, otis_df):
        result = otis_did_policy(otis_df, post_year=2021)
        assert result["n"] > 0
