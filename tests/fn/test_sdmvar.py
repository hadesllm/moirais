"""Tests for moirais.fn.sdmvar."""
import numpy as np
import pytest
from moirais.fn.sdmvar import sdmvar


class TestSdmvar:
    def test_basic(self):
        np.random.seed(33); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); WX=X*0.3; W=np.eye(n)*0.2; rho=0.3; sigma2=1.0
        result = sdmvar(X, WX, W, rho, sigma2)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(33); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); WX=X*0.3; W=np.eye(n)*0.2; rho=0.3; sigma2=1.0
        result = sdmvar(X, WX, W, rho, sigma2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(33); n=15; X=np.column_stack([np.ones(n),np.random.randn(n)]); WX=X*0.3; W=np.eye(n)*0.2; rho=0.3; sigma2=1.0
        result = sdmvar(X, WX, W, rho, sigma2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
