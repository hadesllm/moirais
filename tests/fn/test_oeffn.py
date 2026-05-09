"""Tests for moirais.fn.oeffn — Summary of all effect sizes."""

import numpy as np
from moirais.fn.oeffn import otis_effect_summary


class TestOtisEffectSummary:
    def test_returns_dict(self, otis_df):
        result = otis_effect_summary(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_effect_summary(otis_df)
        for k in ("cohens_d", "point_biserial_r", "n"):
            assert k in result

    def test_d_finite(self, otis_df):
        result = otis_effect_summary(otis_df)
        assert np.isfinite(result["cohens_d"])

    def test_r_bounded(self, otis_df):
        result = otis_effect_summary(otis_df)
        assert -1 <= result["point_biserial_r"] <= 1

    def test_n_matches(self, otis_df):
        result = otis_effect_summary(otis_df)
        assert result["n"] == otis_df[["Y", "D"]].dropna().shape[0]
