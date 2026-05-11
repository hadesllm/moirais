"""Tests for morie.fn.mgwrtvl."""
import numpy as np
import pytest
from morie.fn.mgwrtvl import mgwrtvl


class TestMgwrtvl:
    def test_basic(self):
        np.random.seed(125); coef=np.random.randn(15); se=np.abs(np.random.randn(15))+0.1
        result = mgwrtvl(coef, se)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(125); coef=np.random.randn(15); se=np.abs(np.random.randn(15))+0.1
        result = mgwrtvl(coef, se)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(125); coef=np.random.randn(15); se=np.abs(np.random.randn(15))+0.1
        result = mgwrtvl(coef, se)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
