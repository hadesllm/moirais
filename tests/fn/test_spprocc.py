"""Tests for morie.fn.spprocc."""
import numpy as np
import pytest
from morie.fn.spprocc import spprocc


class TestSpprocc:
    def test_basic(self):
        np.random.seed(151); n=30; y=(np.random.rand(n)>0.5).astype(float); probs=np.clip(y+np.random.randn(n)*0.3,0,1)
        result = spprocc(y, probs)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(151); n=30; y=(np.random.rand(n)>0.5).astype(float); probs=np.clip(y+np.random.randn(n)*0.3,0,1)
        result = spprocc(y, probs)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(151); n=30; y=(np.random.rand(n)>0.5).astype(float); probs=np.clip(y+np.random.randn(n)*0.3,0,1)
        result = spprocc(y, probs)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
