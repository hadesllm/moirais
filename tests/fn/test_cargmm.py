"""Tests for moirais.fn.cargmm."""
import numpy as np
import pytest
from moirais.fn.cargmm import cargmm


class TestCargmm:
    def test_basic(self):
        np.random.seed(23); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = cargmm(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(23); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = cargmm(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(23); n=20; y=np.random.randn(n); W=np.eye(n)*0.2
        result = cargmm(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
