"""Tests for morie.fn.semgmm."""
import numpy as np
import pytest
from morie.fn.semgmm import semgmm


class TestSemgmm:
    def test_basic(self):
        np.random.seed(9); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)
        result = semgmm(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(9); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)
        result = semgmm(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(9); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)
        result = semgmm(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
