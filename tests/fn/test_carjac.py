"""Tests for morie.fn.carjac."""
import numpy as np
import pytest
from morie.fn.carjac import carjac


class TestCarjac:
    def test_basic(self):
        W=np.eye(10)*0.1; rho=0.3
        result = carjac(W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; rho=0.3
        result = carjac(W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; rho=0.3
        result = carjac(W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
