"""Tests for morie.fn.splgtmf."""
import numpy as np
import pytest
from morie.fn.splgtmf import splgtmf


class TestSplgtmf:
    def test_basic(self):
        np.random.seed(153); coef=np.array([0.5,0.3]); rho=0.2; n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splgtmf(coef, rho, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(153); coef=np.array([0.5,0.3]); rho=0.2; n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splgtmf(coef, rho, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(153); coef=np.array([0.5,0.3]); rho=0.2; n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2
        result = splgtmf(coef, rho, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
