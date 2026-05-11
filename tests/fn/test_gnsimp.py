"""Tests for morie.fn.gnsimp."""
import numpy as np
import pytest
from morie.fn.gnsimp import gnsimp


class TestGnsimp:
    def test_basic(self):
        np.random.seed(52); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.2; lam=0.2; W=np.eye(8)*0.1
        result = gnsimp(coef, theta, rho, lam, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(52); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.2; lam=0.2; W=np.eye(8)*0.1
        result = gnsimp(coef, theta, rho, lam, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(52); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); rho=0.2; lam=0.2; W=np.eye(8)*0.1
        result = gnsimp(coef, theta, rho, lam, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
