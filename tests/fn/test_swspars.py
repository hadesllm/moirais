"""Tests for morie.fn.swspars."""
import numpy as np
import pytest
from morie.fn.swspars import swspars


class TestSwspars:
    def test_basic(self):
        W=np.random.rand(8,8)*0.5; thr=0.2
        result = swspars(W, thr)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.random.rand(8,8)*0.5; thr=0.2
        result = swspars(W, thr)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.random.rand(8,8)*0.5; thr=0.2
        result = swspars(W, thr)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
