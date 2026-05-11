"""Tests for morie.fn.sdemimp."""
import numpy as np
import pytest
from morie.fn.sdemimp import sdemimp


class TestSdemimp:
    def test_basic(self):
        np.random.seed(47); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); lam=0.3; W=np.eye(8)*0.1
        result = sdemimp(coef, theta, lam, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(47); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); lam=0.3; W=np.eye(8)*0.1
        result = sdemimp(coef, theta, lam, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(47); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); lam=0.3; W=np.eye(8)*0.1
        result = sdemimp(coef, theta, lam, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
