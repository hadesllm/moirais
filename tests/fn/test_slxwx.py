"""Tests for moirais.fn.slxwx."""
import numpy as np
import pytest
from moirais.fn.slxwx import slxwx


class TestSlxwx:
    def test_basic(self):
        np.random.seed(41); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxwx(X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(41); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxwx(X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(41); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.3
        result = slxwx(X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
