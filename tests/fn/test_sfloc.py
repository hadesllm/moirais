"""Tests for morie.fn.sfloc."""
import numpy as np
import pytest
from morie.fn.sfloc import sfloc


class TestSfloc:
    def test_basic(self):
        np.random.seed(192); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3; i=0
        result = sfloc(y, X, W, i)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(192); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3; i=0
        result = sfloc(y, X, W, i)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(192); n=20; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3; i=0
        result = sfloc(y, X, W, i)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
