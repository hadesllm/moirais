"""Tests for morie.fn.swntri."""
import numpy as np
import pytest
from morie.fn.swntri import swntri


class TestSwntri:
    def test_basic(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swntri(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swntri(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swntri(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
