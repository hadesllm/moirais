"""Tests for morie.fn.gwrtri."""
import numpy as np
import pytest
from morie.fn.gwrtri import gwrtri


class TestGwrtri:
    def test_basic(self):
        np.random.seed(101); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrtri(dists, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(101); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrtri(dists, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(101); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrtri(dists, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
