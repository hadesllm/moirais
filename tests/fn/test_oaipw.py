"""Tests for moirais.fn.oaipw — AIPW doubly-robust estimator."""

import numpy as np
from moirais.fn.oaipw import otis_aipw


class TestOtisAipw:
    def test_returns_dict(self, otis_df):
        result = otis_aipw(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_aipw(otis_df)
        for k in ("ate", "se", "pval", "ci_lower", "ci_upper", "n"):
            assert k in result

    def test_ate_finite(self, otis_df):
        result = otis_aipw(otis_df)
        assert np.isfinite(result["ate"])

    def test_se_positive(self, otis_df):
        result = otis_aipw(otis_df)
        assert result["se"] > 0

    def test_pval_range(self, otis_df):
        result = otis_aipw(otis_df)
        assert 0 <= result["pval"] <= 1
