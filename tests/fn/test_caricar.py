"""Tests for moirais.fn.caricar."""
import numpy as np
import pytest
from moirais.fn.caricar import caricar


class TestCaricar:
    def test_basic(self):
        np.random.seed(24); phi=np.random.randn(10); W=np.eye(10)*0.2
        result = caricar(phi, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(24); phi=np.random.randn(10); W=np.eye(10)*0.2
        result = caricar(phi, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(24); phi=np.random.randn(10); W=np.eye(10)*0.2
        result = caricar(phi, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
