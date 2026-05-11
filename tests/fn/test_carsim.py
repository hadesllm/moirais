"""Tests for morie.fn.carsim."""
import numpy as np
import pytest
from morie.fn.carsim import carsim


class TestCarsim:
    def test_basic(self):
        np.random.seed(28); W=np.eye(10)*0.2; rho=0.3; sigma2=1.0; nsim=9
        result = carsim(W, rho, sigma2, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(28); W=np.eye(10)*0.2; rho=0.3; sigma2=1.0; nsim=9
        result = carsim(W, rho, sigma2, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(28); W=np.eye(10)*0.2; rho=0.3; sigma2=1.0; nsim=9
        result = carsim(W, rho, sigma2, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
