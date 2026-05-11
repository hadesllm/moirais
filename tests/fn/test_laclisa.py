"""Tests for morie.fn.laclisa."""
import numpy as np
import pytest
from morie.fn.laclisa import laclisa


class TestLaclisa:
    def test_basic(self):
        np.random.seed(197); y=np.random.randn(20); W=np.eye(20)*0.3
        result = laclisa(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(197); y=np.random.randn(20); W=np.eye(20)*0.3
        result = laclisa(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(197); y=np.random.randn(20); W=np.eye(20)*0.3
        result = laclisa(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
