"""Tests for moirais.fn.semvar."""
import numpy as np
import pytest
from moirais.fn.semvar import semvar


class TestSemvar:
    def test_basic(self):
        np.random.seed(15); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; lam=0.3; sigma2=1.0
        result = semvar(X, W, lam, sigma2)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(15); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; lam=0.3; sigma2=1.0
        result = semvar(X, W, lam, sigma2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(15); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; lam=0.3; sigma2=1.0
        result = semvar(X, W, lam, sigma2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
