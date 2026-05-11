"""Tests for scpct -- percentile norms."""
import numpy as np
from morie.fn.scpct import percentile_norms
from morie.fn._containers import DescriptiveResult


class TestPercentileNorms:
    def test_basic(self):
        scores = np.arange(1, 101, dtype=float)
        result = percentile_norms(scores)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 9

    def test_median_at_50(self):
        scores = np.arange(1, 101, dtype=float)
        result = percentile_norms(scores, percentiles=[50])
        assert abs(result.value.iloc[0]["score_value"] - 50.5) < 1
