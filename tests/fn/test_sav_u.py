"""Tests for sav_u -- UA AVE."""
import numpy as np
from morie.fn.sav_u import subscale_ua_ave
from morie.fn._containers import ESRes


class TestSavU:
    def test_basic(self, mapq_df):
        result = subscale_ua_ave(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ua_ave(X, items=None)
        assert result.estimate > 0
