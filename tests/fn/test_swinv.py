"""Tests for moirais.fn.swinv."""
import numpy as np
import pytest
from moirais.fn.swinv import swinv


class TestSwinv:
    def test_basic(self):
        np.random.seed(70); coords=np.random.rand(12,2); power=1.0
        result = swinv(coords, power)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(70); coords=np.random.rand(12,2); power=1.0
        result = swinv(coords, power)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(70); coords=np.random.rand(12,2); power=1.0
        result = swinv(coords, power)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
