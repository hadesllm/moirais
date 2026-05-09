"""Tests for moirais.fn.sfmemb."""
import numpy as np
import pytest
from moirais.fn.sfmemb import sfmemb


class TestSfmemb:
    def test_basic(self):
        np.random.seed(190); y=np.random.randn(20); W=np.eye(20)*0.3; alpha=0.05
        result = sfmemb(y, W, alpha)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(190); y=np.random.randn(20); W=np.eye(20)*0.3; alpha=0.05
        result = sfmemb(y, W, alpha)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(190); y=np.random.randn(20); W=np.eye(20)*0.3; alpha=0.05
        result = sfmemb(y, W, alpha)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
