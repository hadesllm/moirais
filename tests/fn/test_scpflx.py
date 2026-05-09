"""Tests for moirais.fn.scpflx."""
import numpy as np
import pytest
from moirais.fn.scpflx import scpflx


class TestScpflx:
    def test_basic(self):
        np.random.seed(170); n=10; T=3; NT=n*T; y=np.random.poisson(2,NT).astype(float); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; unit_id=np.tile(np.arange(n),T)
        result = scpflx(y, X, W, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(170); n=10; T=3; NT=n*T; y=np.random.poisson(2,NT).astype(float); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; unit_id=np.tile(np.arange(n),T)
        result = scpflx(y, X, W, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(170); n=10; T=3; NT=n*T; y=np.random.poisson(2,NT).astype(float); X=np.column_stack([np.ones(NT),np.random.randn(NT)]); W=np.eye(n)*0.2; unit_id=np.tile(np.arange(n),T)
        result = scpflx(y, X, W, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
