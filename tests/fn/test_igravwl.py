"""Tests for moirais.fn.igravwl."""
import numpy as np
import pytest
from moirais.fn.igravwl import igravwl


class TestIgravwl:
    def test_basic(self):
        np.random.seed(176); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10; beta=1.0
        result = igravwl(flows, mass_o, mass_d, dist, beta)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(176); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10; beta=1.0
        result = igravwl(flows, mass_o, mass_d, dist, beta)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(176); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10; beta=1.0
        result = igravwl(flows, mass_o, mass_d, dist, beta)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
