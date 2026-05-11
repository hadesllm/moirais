"""Tests for morie.fn.igravpt."""
import numpy as np
import pytest
from morie.fn.igravpt import igravpt


class TestIgravpt:
    def test_basic(self):
        np.random.seed(180); mass=np.random.rand(10)*1e6; dist=np.random.rand(10)*1000+10
        result = igravpt(mass, dist)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(180); mass=np.random.rand(10)*1e6; dist=np.random.rand(10)*1000+10
        result = igravpt(mass, dist)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(180); mass=np.random.rand(10)*1e6; dist=np.random.rand(10)*1000+10
        result = igravpt(mass, dist)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
