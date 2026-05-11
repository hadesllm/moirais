"""Tests for morie.fn.sppiv."""
import numpy as np
import pytest
from morie.fn.sppiv import sppiv


class TestSppiv:
    def test_basic(self):
        np.random.seed(140); n=12; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); Z=np.random.randn(NT,2); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppiv(y, X, Z, W, time_id, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(140); n=12; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); Z=np.random.randn(NT,2); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppiv(y, X, Z, W, time_id, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(140); n=12; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); Z=np.random.randn(NT,2); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppiv(y, X, Z, W, time_id, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
