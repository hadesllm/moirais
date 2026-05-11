"""Tests for morie.fn.igravrt."""
import numpy as np
import pytest
from morie.fn.igravrt import igravrt


class TestIgravrt:
    def test_basic(self):
        np.random.seed(181); mass=np.random.rand(8)*500+10; dist=np.random.rand(8)*20+1; beta=2.0
        result = igravrt(mass, dist, beta)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(181); mass=np.random.rand(8)*500+10; dist=np.random.rand(8)*20+1; beta=2.0
        result = igravrt(mass, dist, beta)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(181); mass=np.random.rand(8)*500+10; dist=np.random.rand(8)*20+1; beta=2.0
        result = igravrt(mass, dist, beta)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
