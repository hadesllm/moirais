"""Tests for rank_sum_test."""
import numpy as np, pytest
from morie.fn.rnksm import rank_sum_test


class TestRankSum:
    def test_different_groups(self):
        r = rank_sum_test([1, 2, 3], [10, 11, 12])
        assert r.test_name == "Wilcoxon rank-sum test"
        assert r.p_value < 0.05

    def test_same_groups(self):
        r = rank_sum_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert r.p_value > 0.05

    def test_empty(self):
        with pytest.raises(ValueError):
            rank_sum_test([], [1, 2])
