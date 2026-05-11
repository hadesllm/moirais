"""Tests for morie.fn.carsig."""
import numpy as np
import pytest
from morie.fn.carsig import carsig


class TestCarsig:
    def test_basic(self):
        np.random.seed(26); resid=np.random.randn(20); n=20
        result = carsig(resid, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(26); resid=np.random.randn(20); n=20
        result = carsig(resid, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(26); resid=np.random.randn(20); n=20
        result = carsig(resid, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
