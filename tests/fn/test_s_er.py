"""Tests for morie.fn.s_er -- ER subscale reliability."""

import numpy as np
from morie.fn.s_er import subscale_er


class TestSubscaleER:

    def test_returns_all_keys(self, mapq_df):
        result = subscale_er(mapq_df)
        for key in ("alpha", "omega", "cr", "ave", "n_items", "n"):
            assert key in result

    def test_n_matches_data(self, mapq_df):
        result = subscale_er(mapq_df)
        assert result["n"] == len(mapq_df)

    def test_alpha_finite(self, mapq_df):
        result = subscale_er(mapq_df)
        assert np.isfinite(result["alpha"])
