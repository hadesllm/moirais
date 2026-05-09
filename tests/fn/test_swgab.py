"""Tests for moirais.fn.swgab."""
import numpy as np
import pytest
from moirais.fn.swgab import swgab


class TestSwgab:
    def test_basic(self):
        np.random.seed(72); coords=np.random.rand(10,2)
        result = swgab(coords)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(72); coords=np.random.rand(10,2)
        result = swgab(coords)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(72); coords=np.random.rand(10,2)
        result = swgab(coords)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
