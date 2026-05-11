"""Tests for morie.fn.igravnb."""
import numpy as np
import pytest
from morie.fn.igravnb import igravnb


class TestIgravnb:
    def test_basic(self):
        np.random.seed(173); n=10; flows=np.random.negative_binomial(5,0.05,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravnb(flows, mass_o, mass_d, dist)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(173); n=10; flows=np.random.negative_binomial(5,0.05,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravnb(flows, mass_o, mass_d, dist)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(173); n=10; flows=np.random.negative_binomial(5,0.05,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravnb(flows, mass_o, mass_d, dist)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
