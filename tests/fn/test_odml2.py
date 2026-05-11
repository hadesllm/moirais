"""Tests for morie.fn.odml2 — DML for alert treatment effect."""

import numpy as np
from morie.fn.odml2 import otis_dml_alert


class TestOtisDmlAlert:
    def test_returns_dict(self, otis_df):
        result = otis_dml_alert(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_dml_alert(otis_df)
        for k in ("ate", "se", "pval", "ci_lower", "ci_upper", "n", "treatment"):
            assert k in result

    def test_ate_finite(self, otis_df):
        result = otis_dml_alert(otis_df)
        assert np.isfinite(result["ate"])

    def test_pval_range(self, otis_df):
        result = otis_dml_alert(otis_df)
        assert 0 <= result["pval"] <= 1

    def test_custom_treatment(self, otis_df):
        result = otis_dml_alert(otis_df, treatment="alert_suicide_risk")
        assert result["treatment"] == "alert_suicide_risk"
