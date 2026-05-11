"""Tests for morie.fn.sarfilt."""
import numpy as np
import pytest
from morie.fn.sarfilt import sarfilt


class TestSarfilt:
    def test_basic(self):
        np.random.seed(3); y=np.random.randn(20); W=np.eye(20)*0.4; rho=0.3
        result = sarfilt(y, W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(3); y=np.random.randn(20); W=np.eye(20)*0.4; rho=0.3
        result = sarfilt(y, W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(3); y=np.random.randn(20); W=np.eye(20)*0.4; rho=0.3
        result = sarfilt(y, W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
