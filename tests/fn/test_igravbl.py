"""Tests for morie.fn.igravbl."""
import numpy as np
import pytest
from morie.fn.igravbl import igravbl


class TestIgravbl:
    def test_basic(self):
        np.random.seed(179); n=5; flow_matrix=np.random.rand(n,n)+0.1; row_totals=np.ones(n); col_totals=np.ones(n)
        result = igravbl(flow_matrix, row_totals, col_totals)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(179); n=5; flow_matrix=np.random.rand(n,n)+0.1; row_totals=np.ones(n); col_totals=np.ones(n)
        result = igravbl(flow_matrix, row_totals, col_totals)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(179); n=5; flow_matrix=np.random.rand(n,n)+0.1; row_totals=np.ones(n); col_totals=np.ones(n)
        result = igravbl(flow_matrix, row_totals, col_totals)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
