"""Tests for morie.fn.spprgmm."""
import numpy as np
import pytest
from morie.fn.spprgmm import spprgmm


class TestSpprgmm:
    def test_basic(self):
        np.random.seed(148); n=25; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = spprgmm(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(148); n=25; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = spprgmm(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(148); n=25; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = spprgmm(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
