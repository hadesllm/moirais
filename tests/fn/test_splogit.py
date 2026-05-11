"""Tests for morie.fn.splogit."""
import numpy as np
import pytest
from morie.fn.splogit import splogit


class TestSplogit:
    def test_basic(self):
        np.random.seed(145); n=30; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splogit(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(145); n=30; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splogit(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(145); n=30; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splogit(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
