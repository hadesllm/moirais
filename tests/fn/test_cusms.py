"""Tests for morie.fn.cusms -- CUSUM test."""
import numpy as np
import pytest
from morie.fn.cusms import cusum_test


class TestCUSUM:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1, 2]) + rng.normal(0, 0.5, n)
        res = cusum_test(y, X)
        assert "cusum_stat" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            cusum_test(np.ones(2), np.ones((2, 2)))

    def test_cheatsheet(self):
        from morie.fn.cusms import cheatsheet
        assert isinstance(cheatsheet(), str)
