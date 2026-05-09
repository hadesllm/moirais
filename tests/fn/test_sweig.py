"""Tests for moirais.fn.sweig."""
import numpy as np
import pytest
from moirais.fn.sweig import sweig


class TestSweig:
    def test_basic(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = sweig(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = sweig(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
        result = sweig(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
