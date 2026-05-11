"""Tests for morie.fn.sacwald."""
import numpy as np
import pytest
from morie.fn.sacwald import sacwald


class TestSacwald:
    def test_basic(self):
        np.random.seed(59); params=np.array([0.3,0.2]); vcov=np.eye(2)*0.01
        result = sacwald(params, vcov)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(59); params=np.array([0.3,0.2]); vcov=np.eye(2)*0.01
        result = sacwald(params, vcov)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(59); params=np.array([0.3,0.2]); vcov=np.eye(2)*0.01
        result = sacwald(params, vcov)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
