"""Tests for morie.fn.lmhe."""
import numpy as np
import pytest
from morie.fn.lmhe import lmhe


class TestLmhe:
    def test_basic(self):
        np.random.seed(87); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmhe(resid, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(87); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmhe(resid, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(87); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmhe(resid, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
