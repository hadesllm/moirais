"""Tests for morie.fn.swkern."""
import numpy as np
import pytest
from morie.fn.swkern import swkern


class TestSwkern:
    def test_basic(self):
        np.random.seed(68); coords=np.random.rand(12,2); bw=0.5; kernel='gaussian'
        result = swkern(coords, bw, kernel)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(68); coords=np.random.rand(12,2); bw=0.5; kernel='gaussian'
        result = swkern(coords, bw, kernel)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(68); coords=np.random.rand(12,2); bw=0.5; kernel='gaussian'
        result = swkern(coords, bw, kernel)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
