"""Tests for moirais.fn.gwrcv."""
import numpy as np
import pytest
from moirais.fn.gwrcv import gwrcv


class TestGwrcv:
    def test_basic(self):
        np.random.seed(98); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrcv(y, X, coords, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(98); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrcv(y, X, coords, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(98); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrcv(y, X, coords, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
