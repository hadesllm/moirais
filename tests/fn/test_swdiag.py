"""Tests for moirais.fn.swdiag."""
import numpy as np
import pytest
from moirais.fn.swdiag import swdiag


class TestSwdiag:
    def test_basic(self):
        W=np.eye(5)*2+0.1
        result = swdiag(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(5)*2+0.1
        result = swdiag(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(5)*2+0.1
        result = swdiag(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
