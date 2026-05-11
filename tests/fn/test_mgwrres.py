"""Tests for morie.fn.mgwrres."""
import numpy as np
import pytest
from morie.fn.mgwrres import mgwrres


class TestMgwrres:
    def test_basic(self):
        np.random.seed(118); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrres(y, X, coords, bws)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(118); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrres(y, X, coords, bws)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(118); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bws=None
        result = mgwrres(y, X, coords, bws)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
