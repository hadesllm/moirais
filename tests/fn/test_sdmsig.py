"""Tests for morie.fn.sdmsig."""
import numpy as np
import pytest
from morie.fn.sdmsig import sdmsig


class TestSdmsig:
    def test_basic(self):
        np.random.seed(34); resid=np.random.randn(20); n=20
        result = sdmsig(resid, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(34); resid=np.random.randn(20); n=20
        result = sdmsig(resid, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(34); resid=np.random.randn(20); n=20
        result = sdmsig(resid, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
