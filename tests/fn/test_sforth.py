"""Tests for moirais.fn.sforth."""
import numpy as np
import pytest
from moirais.fn.sforth import sforth


class TestSforth:
    def test_basic(self):
        np.random.seed(187); evecs=np.linalg.qr(np.random.randn(10,3))[0]
        result = sforth(evecs)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(187); evecs=np.linalg.qr(np.random.randn(10,3))[0]
        result = sforth(evecs)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(187); evecs=np.linalg.qr(np.random.randn(10,3))[0]
        result = sforth(evecs)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
