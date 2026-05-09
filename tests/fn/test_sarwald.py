"""Tests for moirais.fn.sarwald."""
import numpy as np
import pytest
from moirais.fn.sarwald import sarwald


class TestSarwald:
    def test_basic(self):
        rho=0.4; se_rho=0.1
        result = sarwald(rho, se_rho)
        assert result is not None

    def test_returns_spatial_result(self):
        rho=0.4; se_rho=0.1
        result = sarwald(rho, se_rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        rho=0.4; se_rho=0.1
        result = sarwald(rho, se_rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
