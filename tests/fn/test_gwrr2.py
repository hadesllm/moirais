"""Tests for morie.fn.gwrr2."""
import numpy as np
import pytest
from morie.fn.gwrr2 import gwrr2


class TestGwrr2:
    def test_basic(self):
        np.random.seed(103); y=np.random.randn(20); y_hat=y+np.random.randn(20)*0.1
        result = gwrr2(y, y_hat)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(103); y=np.random.randn(20); y_hat=y+np.random.randn(20)*0.1
        result = gwrr2(y, y_hat)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(103); y=np.random.randn(20); y_hat=y+np.random.randn(20)*0.1
        result = gwrr2(y, y_hat)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
