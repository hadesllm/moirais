"""Tests for morie.fn.sdmolsi."""
import numpy as np
import pytest
from morie.fn.sdmolsi import sdmolsi


class TestSdmolsi:
    def test_basic(self):
        np.random.seed(38); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = sdmolsi(y, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(38); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = sdmolsi(y, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(38); n=30; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = sdmolsi(y, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
