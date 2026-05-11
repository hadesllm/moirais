"""Tests for morie.fn.lacjmc."""
import numpy as np
import pytest
from morie.fn.lacjmc import lacjmc


class TestLacjmc:
    def test_basic(self):
        np.random.seed(196); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3; nsim=9
        result = lacjmc(y_binary, W, nsim)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(196); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3; nsim=9
        result = lacjmc(y_binary, W, nsim)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(196); y_binary=(np.random.rand(20)>0.5).astype(int); W=np.eye(20)*0.3; nsim=9
        result = lacjmc(y_binary, W, nsim)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
