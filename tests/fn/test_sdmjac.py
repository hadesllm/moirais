"""Tests for moirais.fn.sdmjac."""
import numpy as np
import pytest
from moirais.fn.sdmjac import sdmjac


class TestSdmjac:
    def test_basic(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmjac(W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmjac(W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmjac(W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
