"""Tests for moirais.fn.lmdiag."""
import numpy as np
import pytest
from moirais.fn.lmdiag import lmdiag


class TestLmdiag:
    def test_basic(self):
        np.random.seed(85); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmdiag(resid, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(85); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmdiag(resid, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(85); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.5
        result = lmdiag(resid, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
