"""Tests for morie.fn.sdmdet."""
import numpy as np
import pytest
from morie.fn.sdmdet import sdmdet


class TestSdmdet:
    def test_basic(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmdet(W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmdet(W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; rho=0.3
        result = sdmdet(W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
