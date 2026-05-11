"""Tests for morie.fn.igravfe."""
import numpy as np
import pytest
from morie.fn.igravfe import igravfe


class TestIgravfe:
    def test_basic(self):
        np.random.seed(174); n=20; flows=np.random.poisson(50,n).astype(float); origin_id=np.random.randint(0,5,n); dest_id=np.random.randint(0,5,n); dist=np.random.rand(n)*1000+10
        result = igravfe(flows, origin_id, dest_id, dist)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(174); n=20; flows=np.random.poisson(50,n).astype(float); origin_id=np.random.randint(0,5,n); dest_id=np.random.randint(0,5,n); dist=np.random.rand(n)*1000+10
        result = igravfe(flows, origin_id, dest_id, dist)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(174); n=20; flows=np.random.poisson(50,n).astype(float); origin_id=np.random.randint(0,5,n); dest_id=np.random.randint(0,5,n); dist=np.random.rand(n)*1000+10
        result = igravfe(flows, origin_id, dest_id, dist)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
