"""Tests for morie.fn.splgtml."""
import numpy as np
import pytest
from morie.fn.splgtml import splgtml


class TestSplgtml:
    def test_basic(self):
        np.random.seed(152); n=20; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; nsim=9
        result = splgtml(y, X, W, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(152); n=20; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; nsim=9
        result = splgtml(y, X, W, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(152); n=20; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; nsim=9
        result = splgtml(y, X, W, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
