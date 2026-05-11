"""Tests for morie.fn.semres."""
import numpy as np
import pytest
from morie.fn.semres import semres


class TestSemres:
    def test_basic(self):
        np.random.seed(14); resid=np.random.randn(20); W=np.eye(20)*0.3; lam=0.3
        result = semres(resid, W, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(14); resid=np.random.randn(20); W=np.eye(20)*0.3; lam=0.3
        result = semres(resid, W, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(14); resid=np.random.randn(20); W=np.eye(20)*0.3; lam=0.3
        result = semres(resid, W, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
