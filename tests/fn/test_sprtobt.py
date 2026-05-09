"""Tests for moirais.fn.sprtobt."""
import numpy as np
import pytest
from moirais.fn.sprtobt import sprtobt


class TestSprtobt:
    def test_basic(self):
        np.random.seed(156); n=25; y=np.clip(np.random.randn(n),0,None); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprtobt(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(156); n=25; y=np.clip(np.random.randn(n),0,None); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprtobt(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(156); n=25; y=np.clip(np.random.randn(n),0,None); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sprtobt(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
