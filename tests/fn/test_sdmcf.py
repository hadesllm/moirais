"""Tests for morie.fn.sdmcf."""
import numpy as np
import pytest
from morie.fn.sdmcf import sdmcf


class TestSdmcf:
    def test_basic(self):
        np.random.seed(31); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdmcf(coef, theta, vcov)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(31); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdmcf(coef, theta, vcov)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(31); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdmcf(coef, theta, vcov)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
