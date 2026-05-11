"""Tests for morie.fn.carml."""
import numpy as np
import pytest
from morie.fn.carml import carml


class TestCarml:
    def test_basic(self):
        np.random.seed(22); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = carml(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(22); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = carml(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(22); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = carml(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
