"""Tests for morie.fn.semflt."""
import numpy as np
import pytest
from morie.fn.semflt import semflt


class TestSemflt:
    def test_basic(self):
        np.random.seed(16); y=np.random.randn(20); W=np.eye(20)*0.4; lam=0.3
        result = semflt(y, W, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(16); y=np.random.randn(20); W=np.eye(20)*0.4; lam=0.3
        result = semflt(y, W, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(16); y=np.random.randn(20); W=np.eye(20)*0.4; lam=0.3
        result = semflt(y, W, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
