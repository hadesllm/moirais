"""Tests for moirais.fn.sarres."""
import numpy as np
import pytest
from moirais.fn.sarres import sarres


class TestSarres:
    def test_basic(self):
        np.random.seed(4); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = sarres(resid, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(4); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = sarres(resid, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(4); resid=np.random.randn(20); W=np.eye(20)*0.3
        result = sarres(resid, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
