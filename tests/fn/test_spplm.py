"""Tests for moirais.fn.spplm."""
import numpy as np
import pytest
from moirais.fn.spplm import spplm


class TestSpplm:
    def test_basic(self):
        np.random.seed(135); n=15; T=3; resid=np.random.randn(n*T); W=np.eye(n)*0.3; 
        result = spplm(resid, W, n, T)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(135); n=15; T=3; resid=np.random.randn(n*T); W=np.eye(n)*0.3; 
        result = spplm(resid, W, n, T)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(135); n=15; T=3; resid=np.random.randn(n*T); W=np.eye(n)*0.3; 
        result = spplm(resid, W, n, T)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
