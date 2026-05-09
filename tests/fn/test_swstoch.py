"""Tests for moirais.fn.swstoch."""
import numpy as np
import pytest
from moirais.fn.swstoch import swstoch


class TestSwstoch:
    def test_basic(self):
        W=np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],dtype=float)
        result = swstoch(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],dtype=float)
        result = swstoch(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],dtype=float)
        result = swstoch(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
