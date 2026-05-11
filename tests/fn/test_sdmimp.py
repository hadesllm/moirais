"""Tests for morie.fn.sdmimp."""
import numpy as np
import pytest
from morie.fn.sdmimp import sdmimp


class TestSdmimp:
    def test_basic(self):
        np.random.seed(30); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.3; W=np.eye(8)*0.1
        result = sdmimp(coef, theta, rho, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(30); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.3; W=np.eye(8)*0.1
        result = sdmimp(coef, theta, rho, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(30); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.3; W=np.eye(8)*0.1
        result = sdmimp(coef, theta, rho, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
