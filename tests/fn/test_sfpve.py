"""Tests for morie.fn.sfpve."""
import numpy as np
import pytest
from morie.fn.sfpve import sfpve


class TestSfpve:
    def test_basic(self):
        np.random.seed(189); y=np.random.randn(20); evecs=np.linalg.qr(np.random.randn(20,3))[0]
        result = sfpve(y, evecs)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(189); y=np.random.randn(20); evecs=np.linalg.qr(np.random.randn(20,3))[0]
        result = sfpve(y, evecs)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(189); y=np.random.randn(20); evecs=np.linalg.qr(np.random.randn(20,3))[0]
        result = sfpve(y, evecs)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
