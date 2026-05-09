"""Tests for moirais.fn.igraver."""
import numpy as np
import pytest
from moirais.fn.igraver import igraver


class TestIgraver:
    def test_basic(self):
        np.random.seed(177); flows=np.random.poisson(100,20).astype(float); flows_hat=flows+np.random.randn(20)*5
        result = igraver(flows, flows_hat)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(177); flows=np.random.poisson(100,20).astype(float); flows_hat=flows+np.random.randn(20)*5
        result = igraver(flows, flows_hat)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(177); flows=np.random.poisson(100,20).astype(float); flows_hat=flows+np.random.randn(20)*5
        result = igraver(flows, flows_hat)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
