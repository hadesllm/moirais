"""Tests for moirais.fn.igravcl."""
import numpy as np
import pytest
from moirais.fn.igravcl import igravcl


class TestIgravcl:
    def test_basic(self):
        np.random.seed(182); n=15; flows=np.random.poisson(50,n).astype(float); mass_o=np.random.rand(n)*1e5; mass_d=np.random.rand(n)*1e5; dist=np.random.rand(n)*800+10
        result = igravcl(flows, mass_o, mass_d, dist)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(182); n=15; flows=np.random.poisson(50,n).astype(float); mass_o=np.random.rand(n)*1e5; mass_d=np.random.rand(n)*1e5; dist=np.random.rand(n)*800+10
        result = igravcl(flows, mass_o, mass_d, dist)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(182); n=15; flows=np.random.poisson(50,n).astype(float); mass_o=np.random.rand(n)*1e5; mass_d=np.random.rand(n)*1e5; dist=np.random.rand(n)*800+10
        result = igravcl(flows, mass_o, mass_d, dist)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
