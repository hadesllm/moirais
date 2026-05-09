"""Tests for sav_e -- EE AVE."""
import numpy as np
from moirais.fn.sav_e import subscale_ee_ave
from moirais.fn._containers import ESRes


class TestSavE:
    def test_basic(self, mapq_df):
        result = subscale_ee_ave(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ee_ave(X, items=None)
        assert result.estimate > 0
