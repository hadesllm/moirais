"""Tests for moirais.fn.schfd — Schoenfeld test for PH."""

import numpy as np
import pytest

from moirais.fn.schfd import schoenfeld_test


class TestSchoenfeldTest:
    def test_no_violation(self):
        rng = np.random.default_rng(42)
        times = np.sort(rng.exponential(5, 50))
        residuals = rng.normal(0, 1, 50)
        res = schoenfeld_test(residuals, times)
        assert res.p_value > 0.01

    def test_violation(self):
        times = np.arange(1, 51, dtype=float)
        residuals = times * 0.5 + np.random.default_rng(42).normal(0, 0.5, 50)
        res = schoenfeld_test(residuals, times)
        assert res.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            schoenfeld_test([1, 2], [1, 2])
