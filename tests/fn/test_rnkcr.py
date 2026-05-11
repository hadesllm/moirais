"""Tests for rank_correlation_test."""
import numpy as np, pytest
from morie.fn.rnkcr import rank_correlation_test


class TestRankCorrelation:
    def test_spearman_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = rank_correlation_test(x, y, method="spearman")
        assert r.statistic == pytest.approx(1.0)
        assert r.p_value < 0.05

    def test_kendall(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = rank_correlation_test(x, y, method="kendall")
        assert r.statistic == pytest.approx(1.0)

    def test_bad_method(self):
        with pytest.raises(ValueError):
            rank_correlation_test([1, 2, 3], [4, 5, 6], method="bad")
