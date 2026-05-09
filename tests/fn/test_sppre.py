"""Tests for moirais.fn.sppre."""
import numpy as np
import pytest
from moirais.fn.sppre import sppre


class TestSppre:
    def test_basic(self):
        np.random.seed(127); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppre(y, X, W, time_id, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(127); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppre(y, X, W, time_id, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(127); n=20; T=3; NT=n*T; y=np.random.randn(NT); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; time_id=np.repeat(np.arange(T),n); unit_id=np.tile(np.arange(n),T)
        result = sppre(y, X, W, time_id, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
