"""Tests for morie.fn.scdisp."""
import numpy as np
import pytest
from morie.fn.scdisp import scdisp


class TestScdisp:
    def test_basic(self):
        np.random.seed(169); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = scdisp(y, X)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(169); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = scdisp(y, X)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(169); n=25; y=np.random.poisson(2,n).astype(float); X=np.column_stack([np.ones(n),np.random.randn(n)])
        result = scdisp(y, X)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
