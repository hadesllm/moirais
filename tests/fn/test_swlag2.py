"""Tests for moirais.fn.swlag2."""
import numpy as np
import pytest
from moirais.fn.swlag2 import swlag2


class TestSwlag2:
    def test_basic(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag2(W, y)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag2(W, y)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(5)*0.3; y=np.arange(5,dtype=float)
        result = swlag2(W, y)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
