"""Tests for moirais.fn.swdense."""
import numpy as np
import pytest
from moirais.fn.swdense import swdense


class TestSwdense:
    def test_basic(self):
        W=np.array([[0,1,1],[1,0,0],[1,0,0]],dtype=float)
        result = swdense(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,1],[1,0,0],[1,0,0]],dtype=float)
        result = swdense(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,1],[1,0,0],[1,0,0]],dtype=float)
        result = swdense(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
