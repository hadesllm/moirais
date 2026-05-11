"""Tests for morie.fn.slxflt."""
import numpy as np
import pytest
from morie.fn.slxflt import slxflt


class TestSlxflt:
    def test_basic(self):
        np.random.seed(45); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxflt(X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(45); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxflt(X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(45); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxflt(X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
