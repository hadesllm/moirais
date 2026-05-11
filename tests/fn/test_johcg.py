"""Tests for morie.fn.johcg -- Johansen cointegration."""
import numpy as np
import pytest
from morie.fn.johcg import johansen_test


class TestJohansen:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(100))
        Y = np.column_stack([x, 2 * x + rng.normal(0, 0.5, 100)])
        res = johansen_test(Y, lags=1)
        assert len(res.extra["trace_statistics"]) == 2

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            johansen_test(np.ones(50))

    def test_cheatsheet(self):
        from morie.fn.johcg import cheatsheet
        assert isinstance(cheatsheet(), str)
