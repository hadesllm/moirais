"""Tests for morie.fn.swadapt."""
import numpy as np
import pytest
from morie.fn.swadapt import swadapt


class TestSwadapt:
    def test_basic(self):
        np.random.seed(69); coords=np.random.rand(12,2); k=4
        result = swadapt(coords, k)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(69); coords=np.random.rand(12,2); k=4
        result = swadapt(coords, k)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(69); coords=np.random.rand(12,2); k=4
        result = swadapt(coords, k)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
