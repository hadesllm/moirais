"""Tests for moirais.fn.carvar."""
import numpy as np
import pytest
from moirais.fn.carvar import carvar


class TestCarvar:
    def test_basic(self):
        W=np.eye(10)*0.2; rho=0.3; sigma2=1.0
        result = carvar(W, rho, sigma2)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.2; rho=0.3; sigma2=1.0
        result = carvar(W, rho, sigma2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.2; rho=0.3; sigma2=1.0
        result = carvar(W, rho, sigma2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
