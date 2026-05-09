"""Tests for moirais.fn.carbym."""
import numpy as np
import pytest
from moirais.fn.carbym import carbym


class TestCarbym:
    def test_basic(self):
        np.random.seed(25); n=20; y=np.abs(np.random.randn(n)); W=np.eye(n)*0.3
        result = carbym(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(25); n=20; y=np.abs(np.random.randn(n)); W=np.eye(n)*0.3
        result = carbym(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(25); n=20; y=np.abs(np.random.randn(n)); W=np.eye(n)*0.3
        result = carbym(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
