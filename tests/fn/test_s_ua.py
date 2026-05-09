"""Tests for moirais.fn.s_ua -- UA subscale reliability."""

import numpy as np
from moirais.fn.s_ua import subscale_ua


class TestSubscaleUA:

    def test_returns_all_keys(self, mapq_df):
        result = subscale_ua(mapq_df)
        for key in ("alpha", "omega", "cr", "ave", "n_items", "n"):
            assert key in result

    def test_ave_nonnegative(self, mapq_df):
        result = subscale_ua(mapq_df)
        assert result["ave"] >= 0

    def test_omega_in_range(self, mapq_df):
        result = subscale_ua(mapq_df)
        assert 0 <= result["omega"] <= 1
