"""Tests for morie.fn.gwrsur."""
import numpy as np
import pytest
from morie.fn.gwrsur import gwrsur


class TestGwrsur:
    def test_basic(self):
        np.random.seed(112); n=12; ys=[np.random.randn(n),np.random.randn(n)]; X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrsur(ys, X, coords, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(112); n=12; ys=[np.random.randn(n),np.random.randn(n)]; X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrsur(ys, X, coords, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(112); n=12; ys=[np.random.randn(n),np.random.randn(n)]; X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrsur(ys, X, coords, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
