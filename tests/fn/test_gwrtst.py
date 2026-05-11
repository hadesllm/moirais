"""Tests for morie.fn.gwrtst."""
import numpy as np
import pytest
from morie.fn.gwrtst import gwrtst


class TestGwrtst:
    def test_basic(self):
        np.random.seed(108); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5; nsim=9
        result = gwrtst(y, X, coords, bw, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(108); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5; nsim=9
        result = gwrtst(y, X, coords, bw, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(108); n=15; y=np.random.randn(n); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5; nsim=9
        result = gwrtst(y, X, coords, bw, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
