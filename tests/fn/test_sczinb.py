"""Tests for morie.fn.sczinb."""
import numpy as np
import pytest
from morie.fn.sczinb import sczinb


class TestSczinb:
    def test_basic(self):
        np.random.seed(163); n=25; y=np.random.negative_binomial(1,0.5,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sczinb(y, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(163); n=25; y=np.random.negative_binomial(1,0.5,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sczinb(y, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(163); n=25; y=np.random.negative_binomial(1,0.5,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = sczinb(y, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
