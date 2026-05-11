"""Tests for morie.fn.swcond."""
import numpy as np
import pytest
from morie.fn.swcond import swcond


class TestSwcond:
    def test_basic(self):
        W=np.eye(5)+0.1
        result = swcond(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(5)+0.1
        result = swcond(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(5)+0.1
        result = swcond(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
