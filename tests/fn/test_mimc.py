"""Tests for moirais.fn.mimc."""
import numpy as np
import pytest
from moirais.fn.mimc import mimc


class TestMimc:
    def test_basic(self):
        np.random.seed(94); y=np.random.randn(20); W=np.eye(20)*0.3; nsim=9
        result = mimc(y, W, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(94); y=np.random.randn(20); W=np.eye(20)*0.3; nsim=9
        result = mimc(y, W, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(94); y=np.random.randn(20); W=np.eye(20)*0.3; nsim=9
        result = mimc(y, W, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
