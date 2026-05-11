"""Tests for morie.fn.sfmemp."""
import numpy as np
import pytest
from morie.fn.sfmemp import sfmemp


class TestSfmemp:
    def test_basic(self):
        W=np.eye(10)*0.2
        result = sfmemp(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.2
        result = sfmemp(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.2
        result = sfmemp(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
