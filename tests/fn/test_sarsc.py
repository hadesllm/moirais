"""Tests for moirais.fn.sarsc."""
import numpy as np
import pytest
from moirais.fn.sarsc import sarsc


class TestSarsc:
    def test_basic(self):
        np.random.seed(1); resid=np.random.randn(25); W=np.eye(25)*0.5
        result = sarsc(resid, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(1); resid=np.random.randn(25); W=np.eye(25)*0.5
        result = sarsc(resid, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(1); resid=np.random.randn(25); W=np.eye(25)*0.5
        result = sarsc(resid, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
