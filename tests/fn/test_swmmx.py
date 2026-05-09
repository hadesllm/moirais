"""Tests for moirais.fn.swmmx."""
import numpy as np
import pytest
from moirais.fn.swmmx import swmmx


class TestSwmmx:
    def test_basic(self):
        W=np.array([[0,2,1],[2,0,3],[1,3,0]],dtype=float)
        result = swmmx(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,2,1],[2,0,3],[1,3,0]],dtype=float)
        result = swmmx(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,2,1],[2,0,3],[1,3,0]],dtype=float)
        result = swmmx(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
