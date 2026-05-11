"""Tests for morie.fn.lmrerr."""
import numpy as np
import pytest
from morie.fn.lmrerr import lmrerr


class TestLmrerr:
    def test_basic(self):
        np.random.seed(78); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmrerr(resid, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(78); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmrerr(resid, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(78); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmrerr(resid, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
