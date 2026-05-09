"""Tests for moirais.fn.chows -- Chow test."""
import numpy as np
import pytest
from moirais.fn.chows import chow_test


class TestChow:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1, 2]) + rng.normal(0, 0.5, n)
        res = chow_test(y, X, break_point=50)
        assert res.extra["f_statistic"] >= 0

    def test_invalid_break(self):
        with pytest.raises(ValueError):
            chow_test(np.ones(10), np.ones((10, 5)), break_point=1)

    def test_cheatsheet(self):
        from moirais.fn.chows import cheatsheet
        assert isinstance(cheatsheet(), str)
