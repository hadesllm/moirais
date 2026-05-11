"""Tests for morie.fn.sarsim."""
import numpy as np
import pytest
from morie.fn.sarsim import sarsim


class TestSarsim:
    def test_basic(self):
        np.random.seed(5); coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1; nsim=9
        result = sarsim(coef, rho, W, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(5); coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1; nsim=9
        result = sarsim(coef, rho, W, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(5); coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1; nsim=9
        result = sarsim(coef, rho, W, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
