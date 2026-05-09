"""Tests for moirais.fn.bgtet -- Breusch-Godfrey test."""
import numpy as np
import pytest
from moirais.fn.bgtet import bg_test


class TestBG:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1, 2]) + rng.normal(0, 0.5, n)
        res = bg_test(y, X, order=2)
        assert res.extra["lm_statistic"] >= 0

    def test_short_raises(self):
        with pytest.raises(ValueError):
            bg_test(np.ones(3), np.ones((3, 2)), order=2)

    def test_cheatsheet(self):
        from moirais.fn.bgtet import cheatsheet
        assert isinstance(cheatsheet(), str)
