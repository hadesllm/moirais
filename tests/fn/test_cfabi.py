"""Tests for moirais.fn.cfabi -- bifactor CFA."""

import numpy as np
from moirais.fn.cfabi import cfa_bifactor


class TestCfaBifactor:

    def test_returns_fit_indices(self, mapq_df):
        result = cfa_bifactor(mapq_df)
        assert "cfi" in result
        assert "loadings" in result

    def test_has_general_factor(self, mapq_df):
        result = cfa_bifactor(mapq_df)
        assert "General" in result["loadings"]
        assert len(result["loadings"]["General"]) == 20

    def test_rmsea_nonnegative(self, mapq_df):
        result = cfa_bifactor(mapq_df)
        assert result["rmsea"] >= 0
