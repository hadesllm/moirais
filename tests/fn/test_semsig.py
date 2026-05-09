"""Tests for moirais.fn.semsig."""
import numpy as np
import pytest
from moirais.fn.semsig import semsig


class TestSemsig:
    def test_basic(self):
        np.random.seed(19); resid=np.random.randn(20); n=20
        result = semsig(resid, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(19); resid=np.random.randn(20); n=20
        result = semsig(resid, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(19); resid=np.random.randn(20); n=20
        result = semsig(resid, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
