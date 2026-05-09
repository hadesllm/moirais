"""Tests for moirais.fn.swblk."""
import numpy as np
import pytest
from moirais.fn.swblk import swblk


class TestSwblk:
    def test_basic(self):
        groups=np.array([0,0,1,1,2,2,2])
        result = swblk(groups)
        assert result is not None

    def test_returns_spatial_result(self):
        groups=np.array([0,0,1,1,2,2,2])
        result = swblk(groups)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        groups=np.array([0,0,1,1,2,2,2])
        result = swblk(groups)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
