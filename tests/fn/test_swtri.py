"""Tests for morie.fn.swtri."""
import numpy as np
import pytest
from morie.fn.swtri import swtri


class TestSwtri:
    def test_basic(self):
        np.random.seed(71); coords=np.random.rand(10,2)
        result = swtri(coords)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(71); coords=np.random.rand(10,2)
        result = swtri(coords)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(71); coords=np.random.rand(10,2)
        result = swtri(coords)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
