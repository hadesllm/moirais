"""Tests for moirais.fn.swbin."""
import numpy as np
import pytest
from moirais.fn.swbin import swbin


class TestSwbin:
    def test_basic(self):
        W=np.array([[0,0.5,0.3],[0.5,0,0.2],[0.3,0.2,0]],dtype=float)
        result = swbin(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,0.5,0.3],[0.5,0,0.2],[0.3,0.2,0]],dtype=float)
        result = swbin(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,0.5,0.3],[0.5,0,0.2],[0.3,0.2,0]],dtype=float)
        result = swbin(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
