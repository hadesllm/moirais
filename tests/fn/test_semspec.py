"""Tests for moirais.fn.semspec."""
import numpy as np
import pytest
from moirais.fn.semspec import semspec


class TestSemspec:
    def test_basic(self):
        np.random.seed(21); coef_sar=np.array([0.3,0.5]); coef_sem=np.array([0.3,0.5]); vcov=np.eye(2)*0.01
        result = semspec(coef_sar, coef_sem, vcov)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(21); coef_sar=np.array([0.3,0.5]); coef_sem=np.array([0.3,0.5]); vcov=np.eye(2)*0.01
        result = semspec(coef_sar, coef_sem, vcov)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(21); coef_sar=np.array([0.3,0.5]); coef_sem=np.array([0.3,0.5]); vcov=np.eye(2)*0.01
        result = semspec(coef_sar, coef_sem, vcov)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
