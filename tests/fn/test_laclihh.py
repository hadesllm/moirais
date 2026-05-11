"""Tests for morie.fn.laclihh."""
import numpy as np
import pytest
from morie.fn.laclihh import laclihh


class TestLaclihh:
    def test_basic(self):
        np.random.seed(199); y=np.random.randn(20); W=np.eye(20)*0.3; p_thr=0.05
        result = laclihh(y, W, p_thr)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(199); y=np.random.randn(20); W=np.eye(20)*0.3; p_thr=0.05
        result = laclihh(y, W, p_thr)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(199); y=np.random.randn(20); W=np.eye(20)*0.3; p_thr=0.05
        result = laclihh(y, W, p_thr)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
