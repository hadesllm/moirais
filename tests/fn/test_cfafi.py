"""Tests for morie.fn.cfafi -- CFA fit indices for any structure."""

import numpy as np
from morie.fn.cfafi import cfa_fit
from morie.fn._mapq_const import SUBSCALES


class TestCfaFit:

    def test_returns_all_indices(self, mapq_df):
        result = cfa_fit(mapq_df, SUBSCALES)
        for key in ("cfi", "tli", "rmsea", "srmr", "aic", "bic"):
            assert key in result

    def test_aic_bic_finite(self, mapq_df):
        result = cfa_fit(mapq_df, SUBSCALES)
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_custom_structure(self, mapq_df):
        struct = {"F1": ["EE1", "EE2", "EE3"], "F2": ["EA1", "EA2", "EA3"]}
        result = cfa_fit(mapq_df, struct)
        assert result["df"] > 0
