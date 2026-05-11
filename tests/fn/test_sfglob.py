"""Tests for morie.fn.sfglob."""
import numpy as np
import pytest
from morie.fn.sfglob import sfglob


class TestSfglob:
    def test_basic(self):
        np.random.seed(191); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = sfglob(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(191); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = sfglob(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(191); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = sfglob(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
