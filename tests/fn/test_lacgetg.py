"""Tests for moirais.fn.lacgetg."""
import numpy as np
import pytest
from moirais.fn.lacgetg import lacgetg


class TestLacgetg:
    def test_basic(self):
        np.random.seed(204); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetg(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(204); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetg(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(204); y=np.abs(np.random.randn(20))+0.1; W=np.eye(20)*0.3
        result = lacgetg(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
