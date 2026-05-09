"""Tests for moirais.fn.dmtst -- Diebold-Mariano test."""
import numpy as np
import pytest
from moirais.fn.dmtst import dm_test


class TestDM:
    def test_equal_forecasts(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        f = a + rng.normal(0, 0.1, 50)
        res = dm_test(a, f, f)
        assert abs(res.extra["dm_statistic"]) < 0.01

    def test_different_forecasts(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        f1 = a + rng.normal(0, 0.1, 100)
        f2 = a + rng.normal(0, 1.0, 100)
        res = dm_test(a, f1, f2)
        assert "p_value" in res.extra

    def test_cheatsheet(self):
        from moirais.fn.dmtst import cheatsheet
        assert isinstance(cheatsheet(), str)
