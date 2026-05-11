"""Tests for morie.fn.mgwrbk."""
import numpy as np
import pytest
from morie.fn.mgwrbk import mgwrbk


class TestMgwrbk:
    def test_basic(self):
        np.random.seed(123); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); max_iter=3
        result = mgwrbk(y, X, coords, max_iter)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(123); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); max_iter=3
        result = mgwrbk(y, X, coords, max_iter)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(123); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); max_iter=3
        result = mgwrbk(y, X, coords, max_iter)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
