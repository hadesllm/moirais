"""Tests for moirais.fn.sdmconv."""
import numpy as np
import pytest
from moirais.fn.sdmconv import sdmconv


class TestSdmconv:
    def test_basic(self):
        W=np.eye(10)*0.2; rho=0.3
        result = sdmconv(W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.2; rho=0.3
        result = sdmconv(W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.2; rho=0.3
        result = sdmconv(W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
