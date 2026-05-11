"""Tests for morie.fn.igravsq."""
import numpy as np
import pytest
from morie.fn.igravsq import igravsq


class TestIgravsq:
    def test_basic(self):
        np.random.seed(175); n=5; flow_matrix=np.random.poisson(20,(n,n)).astype(float); mass=np.random.rand(n)*1e6; dist_matrix=np.random.rand(n,n)*500+10
        result = igravsq(flow_matrix, mass, dist_matrix)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(175); n=5; flow_matrix=np.random.poisson(20,(n,n)).astype(float); mass=np.random.rand(n)*1e6; dist_matrix=np.random.rand(n,n)*500+10
        result = igravsq(flow_matrix, mass, dist_matrix)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(175); n=5; flow_matrix=np.random.poisson(20,(n,n)).astype(float); mass=np.random.rand(n)*1e6; dist_matrix=np.random.rand(n,n)*500+10
        result = igravsq(flow_matrix, mass, dist_matrix)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
