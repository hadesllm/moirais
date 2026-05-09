"""Tests for moirais.fn.swsph."""
import numpy as np
import pytest
from moirais.fn.swsph import swsph


class TestSwsph:
    def test_basic(self):
        np.random.seed(73); lat=np.random.uniform(43,45,10); lon=np.random.uniform(-80,-78,10); d=300.0
        result = swsph(lat, lon, d)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(73); lat=np.random.uniform(43,45,10); lon=np.random.uniform(-80,-78,10); d=300.0
        result = swsph(lat, lon, d)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(73); lat=np.random.uniform(43,45,10); lon=np.random.uniform(-80,-78,10); d=300.0
        result = swsph(lat, lon, d)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
