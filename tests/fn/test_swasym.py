"""Tests for moirais.fn.swasym."""
import numpy as np
import pytest
from moirais.fn.swasym import swasym


class TestSwasym:
    def test_basic(self):
        np.random.seed(74); W=np.random.rand(8,8)*0.5
        result = swasym(W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(74); W=np.random.rand(8,8)*0.5
        result = swasym(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(74); W=np.random.rand(8,8)*0.5
        result = swasym(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
