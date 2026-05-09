"""Tests for moirais.fn.semboot."""
import numpy as np
import pytest
from moirais.fn.semboot import semboot


class TestSemboot:
    def test_basic(self):
        np.random.seed(17); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n); B=9
        result = semboot(y, X, W, B)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(17); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n); B=9
        result = semboot(y, X, W, B)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(17); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n); B=9
        result = semboot(y, X, W, B)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
