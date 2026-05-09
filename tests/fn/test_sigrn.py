"""Tests for sign_rank_test."""
import numpy as np, pytest
from moirais.fn.sigrn import sign_rank_test


class TestSignRank:
    def test_paired_diff(self):
        x = [10, 12, 14, 16, 18]
        y = [8, 9, 10, 11, 12]
        r = sign_rank_test(x, y)
        assert r.test_name == "Wilcoxon signed-rank test"
        assert r.p_value < 0.1

    def test_no_diff(self):
        x = [1, 2, 3, 4, 5]
        r = sign_rank_test(x, x)
        assert r.p_value >= 0.05

    def test_empty(self):
        with pytest.raises(ValueError):
            sign_rank_test([])
