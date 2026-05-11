"""Tests for sav_r -- ER AVE."""
import numpy as np
from morie.fn.sav_r import subscale_er_ave
from morie.fn._containers import ESRes


class TestSavR:
    def test_basic(self, mapq_df):
        result = subscale_er_ave(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_er_ave(X, items=None)
        assert result.estimate > 0
