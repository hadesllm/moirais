"""Tests for sav_a -- EA AVE."""
import numpy as np
from morie.fn.sav_a import subscale_ea_ave
from morie.fn._containers import ESRes


class TestSavA:
    def test_basic(self, mapq_df):
        result = subscale_ea_ave(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ea_ave(X, items=None)
        assert result.estimate > 0
