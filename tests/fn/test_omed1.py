"""Tests for moirais.fn.omed1 — Mediation analysis."""

import numpy as np
from moirais.fn.omed1 import otis_mediation


class TestOtisMediation:
    def test_returns_dict(self, otis_df):
        result = otis_mediation(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_mediation(otis_df)
        for k in ("total_effect", "direct_effect", "indirect_effect",
                   "sobel_z", "sobel_pval", "prop_mediated", "n"):
            assert k in result

    def test_decomposition(self, otis_df):
        """Total ~ direct + indirect (approximately)."""
        result = otis_mediation(otis_df)
        total = result["total_effect"]
        direct = result["direct_effect"]
        indirect = result["indirect_effect"]
        # Not exact due to rounding, but close
        assert abs(total - (direct + indirect)) < 0.5

    def test_sobel_pval_range(self, otis_df):
        result = otis_mediation(otis_df)
        assert 0 <= result["sobel_pval"] <= 1
