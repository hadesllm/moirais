"""Tests for morie.fn.sppbe."""
import numpy as np
import pytest
from morie.fn.sppbe import sppbe


class TestSppbe:
    def test_basic(self):
        np.random.seed(128); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); unit_id=np.tile(np.arange(n),T)
        result = sppbe(y, X, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(128); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); unit_id=np.tile(np.arange(n),T)
        result = sppbe(y, X, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(128); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); unit_id=np.tile(np.arange(n),T)
        result = sppbe(y, X, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
