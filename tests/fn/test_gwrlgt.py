"""Tests for moirais.fn.gwrlgt."""
import numpy as np
import pytest
from moirais.fn.gwrlgt import gwrlgt


class TestGwrlgt:
    def test_basic(self):
        np.random.seed(109); n=15; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrlgt(y, X, coords, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(109); n=15; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrlgt(y, X, coords, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(109); n=15; y=(np.random.rand(n)>0.5).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)]); coords=np.random.rand(n,2); bw=0.5
        result = gwrlgt(y, X, coords, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
