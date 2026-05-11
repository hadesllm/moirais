"""Tests for morie.fn.swknn."""
import numpy as np
import pytest
from morie.fn.swknn import swknn


class TestSwknn:
    def test_basic(self):
        np.random.seed(66); coords=np.random.rand(15,2); k=3
        result = swknn(coords, k)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(66); coords=np.random.rand(15,2); k=3
        result = swknn(coords, k)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(66); coords=np.random.rand(15,2); k=3
        result = swknn(coords, k)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
