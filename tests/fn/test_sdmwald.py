"""Tests for morie.fn.sdmwald."""
import numpy as np
import pytest
from morie.fn.sdmwald import sdmwald


class TestSdmwald:
    def test_basic(self):
        rho=0.4; se_rho=0.1
        result = sdmwald(rho, se_rho)
        assert result is not None

    def test_returns_spatial_result(self):
        rho=0.4; se_rho=0.1
        result = sdmwald(rho, se_rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        rho=0.4; se_rho=0.1
        result = sdmwald(rho, se_rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
