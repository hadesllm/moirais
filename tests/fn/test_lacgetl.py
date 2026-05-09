"""Tests for moirais.fn.lacgetl."""
import numpy as np
import pytest
from moirais.fn.lacgetl import lacgetl


class TestLacgetl:
    def test_basic(self):
        np.random.seed(205); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetl(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(205); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetl(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(205); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetl(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
