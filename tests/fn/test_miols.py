"""Tests for morie.fn.miols."""
import numpy as np
import pytest
from morie.fn.miols import miols


class TestMiols:
    def test_basic(self):
        np.random.seed(91); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = miols(resid, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(91); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = miols(resid, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(91); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = miols(resid, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
