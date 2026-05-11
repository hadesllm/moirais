"""Tests for morie.fn.sacrob."""
import numpy as np
import pytest
from morie.fn.sacrob import sacrob


class TestSacrob:
    def test_basic(self):
        np.random.seed(65); n=20; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sacrob(resid, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(65); n=20; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sacrob(resid, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(65); n=20; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sacrob(resid, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
