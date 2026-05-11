"""Tests for morie.fn.sarimp."""
import numpy as np
import pytest
from morie.fn.sarimp import sarimp


class TestSarimp:
    def test_basic(self):
        coef=np.array([1.0,0.5]); rho=0.3; W=np.eye(10)*0.1
        result = sarimp(coef, rho, W)
        assert result is not None

    def test_returns_spatial_result(self):
        coef=np.array([1.0,0.5]); rho=0.3; W=np.eye(10)*0.1
        result = sarimp(coef, rho, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        coef=np.array([1.0,0.5]); rho=0.3; W=np.eye(10)*0.1
        result = sarimp(coef, rho, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
