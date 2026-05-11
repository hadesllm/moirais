"""Tests for morie.fn.swpath."""
import numpy as np
import pytest
from morie.fn.swpath import swpath


class TestSwpath:
    def test_basic(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); i=0; j=2
        result = swpath(W, i, j)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); i=0; j=2
        result = swpath(W, i, j)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float); i=0; j=2
        result = swpath(W, i, j)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
