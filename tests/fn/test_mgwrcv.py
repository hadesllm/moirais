"""Tests for moirais.fn.mgwrcv."""
import numpy as np
import pytest
from moirais.fn.mgwrcv import mgwrcv


class TestMgwrcv:
    def test_basic(self):
        np.random.seed(122); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrcv(y, X, coords, bws)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(122); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrcv(y, X, coords, bws)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(122); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrcv(y, X, coords, bws)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
