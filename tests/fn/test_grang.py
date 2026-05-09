"""Tests for moirais.fn.grang -- Granger causality."""
import numpy as np
import pytest
from moirais.fn.grang import granger_test


class TestGranger:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = np.roll(x, 2) + rng.normal(0, 0.5, 100)
        res = granger_test(y, x, max_lag=4)
        assert res.extra["f_statistic"] >= 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            granger_test(np.ones(10), np.ones(15))

    def test_cheatsheet(self):
        from moirais.fn.grang import cheatsheet
        assert isinstance(cheatsheet(), str)
