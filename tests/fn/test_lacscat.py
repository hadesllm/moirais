"""Tests for moirais.fn.lacscat."""
import numpy as np
import pytest
from moirais.fn.lacscat import lacscat


class TestLacscat:
    def test_basic(self):
        np.random.seed(206); y=np.random.randn(20); W=np.eye(20)*0.3
        result = lacscat(y, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(206); y=np.random.randn(20); W=np.eye(20)*0.3
        result = lacscat(y, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(206); y=np.random.randn(20); W=np.eye(20)*0.3
        result = lacscat(y, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
