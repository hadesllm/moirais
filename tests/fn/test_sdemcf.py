"""Tests for moirais.fn.sdemcf."""
import numpy as np
import pytest
from moirais.fn.sdemcf import sdemcf


class TestSdemcf:
    def test_basic(self):
        np.random.seed(50); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdemcf(coef, theta, vcov)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(50); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdemcf(coef, theta, vcov)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(50); coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2]); vcov=np.eye(4)*0.01
        result = sdemcf(coef, theta, vcov)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
