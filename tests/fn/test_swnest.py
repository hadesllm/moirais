"""Tests for morie.fn.swnest."""
import numpy as np
import pytest
from morie.fn.swnest import swnest


class TestSwnest:
    def test_basic(self):
        W_local=np.eye(6)*0.3; W_global=np.ones((6,6))*0.1
        result = swnest(W_local, W_global)
        assert result is not None

    def test_returns_spatial_result(self):
        W_local=np.eye(6)*0.3; W_global=np.ones((6,6))*0.1
        result = swnest(W_local, W_global)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W_local=np.eye(6)*0.3; W_global=np.ones((6,6))*0.1
        result = swnest(W_local, W_global)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
