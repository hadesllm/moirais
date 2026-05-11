"""Tests for morie.fn.swlag."""
import numpy as np
import pytest
from morie.fn.swlag import swlag


class TestSwlag:
    def test_basic(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag(W, y)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag(W, y)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag(W, y)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
