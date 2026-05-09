"""Tests for scr_a -- EA composite reliability."""
import numpy as np
from moirais.fn.scr_a import subscale_ea_composite_rel
from moirais.fn._containers import ESRes


class TestScrA:
    def test_basic(self, mapq_df):
        result = subscale_ea_composite_rel(mapq_df)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_array_input(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 5))
        result = subscale_ea_composite_rel(X, items=None)
        assert result.estimate > 0
