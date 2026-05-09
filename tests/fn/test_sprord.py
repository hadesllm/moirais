"""Tests for moirais.fn.sprord."""
import numpy as np
import pytest
from moirais.fn.sprord import sprord


class TestSprord:
    def test_basic(self):
        np.random.seed(155); n=25; y=np.random.randint(0,3,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprord(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(155); n=25; y=np.random.randint(0,3,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprord(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(155); n=25; y=np.random.randint(0,3,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprord(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
