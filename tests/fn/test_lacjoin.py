"""Tests for morie.fn.lacjoin."""
import numpy as np
import pytest
from morie.fn.lacjoin import lacjoin


class TestLacjoin:
    def test_basic(self):
        np.random.seed(195); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3
        result = lacjoin(y_binary, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(195); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3
        result = lacjoin(y_binary, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(195); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3
        result = lacjoin(y_binary, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
