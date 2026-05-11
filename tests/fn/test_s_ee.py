"""Tests for morie.fn.s_ee -- EE subscale reliability."""

import numpy as np
from morie.fn.s_ee import subscale_ee


class TestSubscaleEE:

    def test_returns_all_keys(self, mapq_df):
        result = subscale_ee(mapq_df)
        for key in ("alpha", "omega", "cr", "ave", "n_items", "n"):
            assert key in result

    def test_alpha_in_range(self, mapq_df):
        result = subscale_ee(mapq_df)
        assert -1 <= result["alpha"] <= 1

    def test_n_items_is_five(self, mapq_df):
        result = subscale_ee(mapq_df)
        assert result["n_items"] == 5
