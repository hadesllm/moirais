"""Tests for moirais.fn.scpmlm."""
import numpy as np
import pytest
from moirais.fn.scpmlm import scpmlm


class TestScpmlm:
    def test_basic(self):
        np.random.seed(164); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpmlm(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(164); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpmlm(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(164); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpmlm(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
