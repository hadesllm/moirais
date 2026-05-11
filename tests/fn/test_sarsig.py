"""Tests for morie.fn.sarsig."""
import numpy as np
import pytest
from morie.fn.sarsig import sarsig


class TestSarsig:
    def test_basic(self):
        np.random.seed(7); resid=np.random.randn(20); n=20
        result = sarsig(resid, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(7); resid=np.random.randn(20); n=20
        result = sarsig(resid, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(7); resid=np.random.randn(20); n=20
        result = sarsig(resid, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
