"""Tests for moirais.fn.scpois."""
import numpy as np
import pytest
from moirais.fn.scpois import scpois


class TestScpois:
    def test_basic(self):
        np.random.seed(160); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpois(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(160); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpois(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(160); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = scpois(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
