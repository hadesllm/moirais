"""Tests for moirais.fn.s_ea -- EA subscale reliability."""

import numpy as np
from moirais.fn.s_ea import subscale_ea


class TestSubscaleEA:

    def test_returns_all_keys(self, mapq_df):
        result = subscale_ea(mapq_df)
        for key in ("alpha", "omega", "cr", "ave", "n_items", "n"):
            assert key in result

    def test_cr_positive(self, mapq_df):
        result = subscale_ea(mapq_df)
        assert result["cr"] > 0

    def test_custom_items(self, mapq_df):
        result = subscale_ea(mapq_df, items=["EA1", "EA2", "EA3"])
        assert result["n_items"] == 3
