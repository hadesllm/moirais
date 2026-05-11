"""Tests for morie.fn.semrlm."""
import numpy as np
import pytest
from morie.fn.semrlm import semrlm


class TestSemrlm:
    def test_basic(self):
        np.random.seed(12); n=25; resid=np.random.randn(n); W=np.eye(n)*0.5; X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = semrlm(resid, W, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(12); n=25; resid=np.random.randn(n); W=np.eye(n)*0.5; X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = semrlm(resid, W, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(12); n=25; resid=np.random.randn(n); W=np.eye(n)*0.5; X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = semrlm(resid, W, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
