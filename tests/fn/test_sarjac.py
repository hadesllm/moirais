"""Tests for moirais.fn.sarjac."""
import numpy as np
import pytest
from moirais.fn.sarjac import sarjac


class TestSarjac:
    def test_basic(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sarjac(W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sarjac(W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sarjac(W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
