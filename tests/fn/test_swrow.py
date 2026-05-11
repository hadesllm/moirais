"""Tests for morie.fn.swrow."""
import numpy as np
import pytest
from morie.fn.swrow import swrow


class TestSwrow:
    def test_basic(self):
        W=np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=float)
        result = swrow(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=float)
        result = swrow(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=float)
        result = swrow(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
