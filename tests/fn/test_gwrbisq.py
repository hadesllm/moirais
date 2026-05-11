"""Tests for morie.fn.gwrbisq."""
import numpy as np
import pytest
from morie.fn.gwrbisq import gwrbisq


class TestGwrbisq:
    def test_basic(self):
        np.random.seed(100); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrbisq(dists, bw)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(100); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrbisq(dists, bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(100); dists=np.random.rand(10)*0.4; bw=0.5
        result = gwrbisq(dists, bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
