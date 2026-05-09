"""Tests for moirais.fn.lmkoenk."""
import numpy as np
import pytest
from moirais.fn.lmkoenk import lmkoenk


class TestLmkoenk:
    def test_basic(self):
        np.random.seed(89); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmkoenk(resid, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(89); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmkoenk(resid, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(89); n=25; resid=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = lmkoenk(resid, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
