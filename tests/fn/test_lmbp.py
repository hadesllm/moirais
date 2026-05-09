"""Tests for moirais.fn.lmbp."""
import numpy as np
import pytest
from moirais.fn.lmbp import lmbp


class TestLmbp:
    def test_basic(self):
        np.random.seed(88); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmbp(resid, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(88); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmbp(resid, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(88); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmbp(resid, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
