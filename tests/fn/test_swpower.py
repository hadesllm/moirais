"""Tests for morie.fn.swpower."""
import numpy as np
import pytest
from morie.fn.swpower import swpower


class TestSwpower:
    def test_basic(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); p=2
        result = swpower(W, p)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); p=2
        result = swpower(W, p)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); p=2
        result = swpower(W, p)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
