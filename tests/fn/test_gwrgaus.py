"""Tests for morie.fn.gwrgaus."""
import numpy as np
import pytest
from morie.fn.gwrgaus import gwrgaus


class TestGwrgaus:
    def test_basic(self):
        np.random.seed(99); dists=np.random.rand(10)*2; bw=0.5
        result = gwrgaus(dists, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(99); dists=np.random.rand(10)*2; bw=0.5
        result = gwrgaus(dists, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(99); dists=np.random.rand(10)*2; bw=0.5
        result = gwrgaus(dists, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
