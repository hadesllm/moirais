"""Tests for morie.fn.recrd -- Recrudescence probability."""

import numpy as np
import pytest
from morie.fn.recrd import recrd


class TestRecrd:
    def test_all_matching(self):
        pre = np.array([1, 2, 3, 4])
        post = np.array([1, 2, 3, 4])
        pf = np.array([0.1, 0.1, 0.1, 0.1])
        res = recrd(pre, post, allele_freqs=pf)
        assert res.statistic > 0.9

    def test_none_matching(self):
        pre = np.array([1, 2, 3])
        post = np.array([4, 5, 6])
        pf = np.array([0.1, 0.1, 0.1])
        res = recrd(pre, post, allele_freqs=pf)
        assert res.statistic < 0.1

    def test_probabilities_sum_to_one(self):
        pre = np.array([1, 2])
        post = np.array([1, 5])
        res = recrd(pre, post)
        assert abs(res.statistic + res.p_value - 1.0) < 1e-10

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            recrd(np.array([1, 2]), np.array([1]))
