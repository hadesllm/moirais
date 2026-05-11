"""Tests for morie.fn.scpboot."""
import numpy as np
import pytest
from morie.fn.scpboot import scpboot


class TestScpboot:
    def test_basic(self):
        np.random.seed(168); n=20; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; B=9
        result = scpboot(y, X, W, B)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(168); n=20; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; B=9
        result = scpboot(y, X, W, B)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(168); n=20; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; B=9
        result = scpboot(y, X, W, B)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
