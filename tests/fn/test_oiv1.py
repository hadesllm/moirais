"""Tests for morie.fn.oiv1 — IV estimation (2SLS)."""

import numpy as np
from morie.fn.oiv1 import otis_iv_distance


class TestOtisIvDistance:
    def test_returns_dict(self, otis_df):
        result = otis_iv_distance(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_iv_distance(otis_df)
        for k in ("iv_estimate", "se", "pval", "n", "first_stage_f", "instrument"):
            assert k in result

    def test_iv_finite(self, otis_df):
        result = otis_iv_distance(otis_df)
        assert np.isfinite(result["iv_estimate"])

    def test_default_instrument(self, otis_df):
        result = otis_iv_distance(otis_df)
        assert result["instrument"] == "sentence_days"

    def test_custom_instrument(self, otis_df):
        result = otis_iv_distance(otis_df, instrument="alert_mental_health")
        assert result["instrument"] == "alert_mental_health"
