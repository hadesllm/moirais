"""Tests for moirais.fn.sfmi."""
import numpy as np
import pytest
from moirais.fn.sfmi import sfmi


class TestSfmi:
    def test_basic(self):
        np.random.seed(188); resid_f=np.random.randn(20); W=np.eye(20)*0.2
        result = sfmi(resid_f, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(188); resid_f=np.random.randn(20); W=np.eye(20)*0.2
        result = sfmi(resid_f, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(188); resid_f=np.random.randn(20); W=np.eye(20)*0.2
        result = sfmi(resid_f, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
