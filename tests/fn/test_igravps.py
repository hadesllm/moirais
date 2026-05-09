"""Tests for moirais.fn.igravps."""
import numpy as np
import pytest
from moirais.fn.igravps import igravps


class TestIgravps:
    def test_basic(self):
        np.random.seed(172); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravps(flows, mass_o, mass_d, dist)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(172); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravps(flows, mass_o, mass_d, dist)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(172); n=10; flows=np.random.poisson(100,n).astype(float); mass_o=np.random.rand(n)*1e6; mass_d=np.random.rand(n)*1e6; dist=np.random.rand(n)*1000+10
        result = igravps(flows, mass_o, mass_d, dist)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
