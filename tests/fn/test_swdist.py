"""Tests for moirais.fn.swdist."""
import numpy as np
import pytest
from moirais.fn.swdist import swdist


class TestSwdist:
    def test_basic(self):
        np.random.seed(67); coords=np.random.rand(15,2); d=0.5
        result = swdist(coords, d)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(67); coords=np.random.rand(15,2); d=0.5
        result = swdist(coords, d)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(67); coords=np.random.rand(15,2); d=0.5
        result = swdist(coords, d)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
