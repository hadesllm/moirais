"""Tests for morie.fn.igravvf."""
import numpy as np
import pytest
from morie.fn.igravvf import igravvf


class TestIgravvf:
    def test_basic(self):
        np.random.seed(183); flows_hat=np.abs(np.random.randn(15))*10+1; phi=1.0
        result = igravvf(flows_hat, phi)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(183); flows_hat=np.abs(np.random.randn(15))*10+1; phi=1.0
        result = igravvf(flows_hat, phi)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(183); flows_hat=np.abs(np.random.randn(15))*10+1; phi=1.0
        result = igravvf(flows_hat, phi)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
