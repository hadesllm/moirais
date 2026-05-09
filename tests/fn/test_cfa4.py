"""Tests for moirais.fn.cfa4 -- 4-factor CFA."""

import numpy as np
from moirais.fn.cfa4 import cfa_4factor


class TestCfa4Factor:

    def test_returns_fit_indices(self, mapq_df):
        result = cfa_4factor(mapq_df)
        for key in ("cfi", "tli", "rmsea", "srmr", "chi2", "df", "p_value"):
            assert key in result, f"Missing key: {key}"

    def test_cfi_in_range(self, mapq_df):
        result = cfa_4factor(mapq_df)
        assert 0 <= result["cfi"] <= 1

    def test_loadings_has_four_factors(self, mapq_df):
        result = cfa_4factor(mapq_df)
        assert set(result["loadings"].keys()) == {"EE", "EA", "UA", "ER"}
        for fname, items in result["loadings"].items():
            assert len(items) == 5
