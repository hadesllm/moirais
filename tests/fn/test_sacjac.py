"""Tests for morie.fn.sacjac."""
import numpy as np
import pytest
from morie.fn.sacjac import sacjac


class TestSacjac:
    def test_basic(self):
        W=np.eye(10)*0.1; rho=0.3; lam=0.2
        result = sacjac(W, rho, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; rho=0.3; lam=0.2
        result = sacjac(W, rho, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; rho=0.3; lam=0.2
        result = sacjac(W, rho, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
