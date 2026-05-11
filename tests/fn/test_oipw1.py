"""Tests for morie.fn.oipw1 — IPW for placement effect."""

import numpy as np
from morie.fn.oipw1 import otis_ipw_placement


class TestOtisIpwPlacement:
    def test_returns_dict(self, otis_df):
        result = otis_ipw_placement(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_ipw_placement(otis_df)
        for k in ("ate", "se", "pval", "ci_lower", "ci_upper", "n", "ess_treated", "ess_control"):
            assert k in result

    def test_ate_finite(self, otis_df):
        result = otis_ipw_placement(otis_df)
        assert np.isfinite(result["ate"])

    def test_ess_positive(self, otis_df):
        result = otis_ipw_placement(otis_df)
        assert result["ess_treated"] > 0
        assert result["ess_control"] > 0
