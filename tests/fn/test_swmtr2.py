"""Tests for morie.fn.swmtr2."""
import numpy as np
import pytest
from morie.fn.swmtr2 import swmtr2


class TestSwmtr2:
    def test_basic(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swmtr2(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swmtr2(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = swmtr2(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
