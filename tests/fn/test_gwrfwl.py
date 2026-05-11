"""Tests for morie.fn.gwrfwl."""
import numpy as np
import pytest
from morie.fn.gwrfwl import gwrfwl


class TestGwrfwl:
    def test_basic(self):
        np.random.seed(111); n=15; y=np.random.randn(n); X1=np.column_stack([np.ones(n),np.random.randn(n)]); X2=np.random.randn(n,1); coords=np.random.rand(n,2); bw=0.5
        result = gwrfwl(y, X1, X2, coords, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(111); n=15; y=np.random.randn(n); X1=np.column_stack([np.ones(n),np.random.randn(n)]); X2=np.random.randn(n,1); coords=np.random.rand(n,2); bw=0.5
        result = gwrfwl(y, X1, X2, coords, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(111); n=15; y=np.random.randn(n); X1=np.column_stack([np.ones(n),np.random.randn(n)]); X2=np.random.randn(n,1); coords=np.random.rand(n,2); bw=0.5
        result = gwrfwl(y, X1, X2, coords, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
