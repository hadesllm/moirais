"""Tests for morie.fn.mgwrbw."""
import numpy as np
import pytest
from morie.fn.mgwrbw import mgwrbw


class TestMgwrbw:
    def test_basic(self):
        np.random.seed(115); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrbw(y, X, coords)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(115); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrbw(y, X, coords)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(115); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2)
        result = mgwrbw(y, X, coords)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
