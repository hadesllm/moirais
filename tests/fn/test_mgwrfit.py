"""Tests for morie.fn.mgwrfit."""
import numpy as np
import pytest
from morie.fn.mgwrfit import mgwrfit


class TestMgwrfit:
    def test_basic(self):
        np.random.seed(114); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrfit(y, X, coords)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(114); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrfit(y, X, coords)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(114); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrfit(y, X, coords)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
